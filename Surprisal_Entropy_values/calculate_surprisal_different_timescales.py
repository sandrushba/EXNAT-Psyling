"""
Calculate surprisal and entropy for texts coming from EXNAT_3 (10 unique texts, 300 words each)
Two timescales: 2-word local context vs. full preceding text context
"""
import os
import re
import ast
import string
import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyprojroot.here import here

tok = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2")
model.eval()
device = torch.device("cpu")
model.to(device)

MAX_CTX_TOKENS = model.config.n_positions - 50  # safety margin below 1024

@torch.no_grad()
def surprisal_entropy_from_ids(ctx_ids, target_word, model, tok, device):
    if len(ctx_ids) > MAX_CTX_TOKENS:
        ctx_ids = ctx_ids[-MAX_CTX_TOKENS:]  # truncate from the left if context too long

    ctx = torch.tensor([ctx_ids], device=device)
    logits_ctx = model(input_ids=ctx).logits[:, -1, :]
    probs_ctx = torch.softmax(logits_ctx, dim=-1)
    log_probs_ctx = torch.log_softmax(logits_ctx, dim=-1)
    entropy_token = float((-probs_ctx * log_probs_ctx).sum().item())

    target_text = " " + target_word
    tgt_ids = tok.encode(target_text, add_special_tokens=False)

    total_logprob = 0.0
    running_ids = ctx_ids[:]
    for tid in tgt_ids:
        x = torch.tensor([running_ids], device=device)
        logits = model(input_ids=x).logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        lp = float(log_probs[0, tid].item())
        total_logprob += lp
        running_ids.append(tid)

    surprisal = -total_logprob
    return surprisal, entropy_token

def surprisal_entropy_two_word_context(context_words, target_word, model, tok, device):
    parts = context_words.strip().split()
    if len(parts) != 2:
        warnings.warn(f"context_words contains {len(parts)} words instead of 2: '{context_words}'")
    context = " ".join(parts)
    ctx_ids = tok.encode(context, add_special_tokens=False)
    return surprisal_entropy_from_ids(ctx_ids, target_word, model, tok, device)

def surprisal_entropy_full_context(preceding_words, target_word, model, tok, device):
    # preceding_words: list of all words from start of the text up to (not including) target
    context = " ".join(preceding_words)
    ctx_ids = tok.encode(context, add_special_tokens=False)
    return surprisal_entropy_from_ids(ctx_ids, target_word, model, tok, device)

base_dir = os.path.dirname(__file__)
rel_path = os.path.join(base_dir, 'EXNAT3_texts_MC_Qs.py')
with open(rel_path, 'r', encoding='utf-8') as f:
    file_content = f.read()

matches = re.findall(r'(?<!pseudo_)text_(\d+)\s*=\s*(\[[^\]]+\])', file_content)
texts = {}
for name, list_str in matches:
    texts[name] = ast.literal_eval(list_str)

rows = []
for text_nr, words in texts.items():
    text_id = f"text_{int(text_nr):02d}"
    for idx, word in enumerate(words):
        clean_word = word.translate(str.maketrans('', '', string.punctuation))
        rows.append({'text_nr': text_id, 'trial_nr': idx + 1, 'word': word, 'word_no_punct': clean_word})
texts_df = pd.DataFrame(rows)

texts_df['surprisal_2_words_context'] = None
texts_df['entropy_2_words_context'] = None
texts_df['surprisal_full_context'] = None
texts_df['entropy_full_context'] = None

n_context_words = 2
previous_text_nr = texts_df["text_nr"][0]

for word_idx, word in enumerate(texts_df["word_no_punct"]):
    if texts_df["text_nr"][word_idx] != previous_text_nr:
        previous_text_nr = texts_df["text_nr"][word_idx]
        if texts_df["trial_nr"][word_idx] <= n_context_words:
            continue

    if texts_df["trial_nr"][word_idx] > n_context_words:
        print(word_idx, texts_df["trial_nr"][word_idx], word)

        two_word_context = ' '.join(texts_df["word_no_punct"][word_idx - n_context_words:word_idx])
        s2, e2 = surprisal_entropy_two_word_context(two_word_context, word, model, tok, device)
        texts_df.at[word_idx, 'surprisal_2_words_context'] = s2
        texts_df.at[word_idx, 'entropy_2_words_context'] = e2

        # full preceding context: from start of THIS text up to current word
        text_start_idx = texts_df.index[texts_df["text_nr"] == previous_text_nr][0]
        full_preceding_words = list(texts_df["word_no_punct"][text_start_idx:word_idx])
        sf, ef = surprisal_entropy_full_context(full_preceding_words, word, model, tok, device)
        texts_df.at[word_idx, 'surprisal_full_context'] = sf
        texts_df.at[word_idx, 'entropy_full_context'] = ef

texts_df.to_csv(here("surprisal_scores/2words_vs_fullcontext_surprisal_entropy_10Texts_08_2026.csv"), encoding="utf-8-sig")
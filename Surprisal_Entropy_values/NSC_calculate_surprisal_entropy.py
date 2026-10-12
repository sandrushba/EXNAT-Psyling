"""
Calculate GPT-2 surprisal and next-token entropy for words stored in a CSV/TSV file.

Expected input columns:
    code    token    text_nr    item

For example:
    1.1.word    If      1    1
    1.2.word    you     1    2
    1.3.word    were    1    3

Two context timescales are calculated:
1. The preceding two words.
2. The complete preceding context within the current text.

The first two words of each text receive missing values because they do not
have two preceding context words. This also makes the two context conditions
directly comparable at the same word positions.

Surprisal and entropy are returned in nats because natural logarithms are used.

Note:
- Surprisal is summed across all GPT-2 subword tokens belonging to the target
  word.
- Entropy is the entropy of GPT-2's next-token distribution immediately before
  the target word.
"""

from pathlib import Path
import warnings
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "dbmdz/german-gpt2"

# Change these filenames/paths as needed.
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "nsc_words.csv"
OUTPUT_FILE = BASE_DIR / "NSC_2words_fullcontext_surprisal_entropy.csv"

N_LOCAL_CONTEXT_WORDS = 2

# Use CUDA when available. Set this explicitly to torch.device("cpu") if needed.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Load GPT-2
# ---------------------------------------------------------------------------

print(f"Loading model: {MODEL_NAME}")
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# German GPT-2 normally uses a maximum context of 1,024 subword tokens.
model_max_positions = getattr(
    model.config,
    "n_positions",
    getattr(model.config, "max_position_embeddings", 1024),
)

# Leave some space for a target word that consists of several subword tokens.
MAX_CONTEXT_TOKENS = model_max_positions - 50


# ---------------------------------------------------------------------------
# Surprisal and entropy functions
# ---------------------------------------------------------------------------

@torch.inference_mode()
def surprisal_entropy_from_ids(context_ids, target_word):
    """
    Calculate target-word surprisal and next-token entropy.

    Parameters
    ----------
    context_ids : list[int]
        GPT-2 token IDs for the preceding context.
    target_word : str
        Current target word.

    Returns
    -------
    surprisal : float
        Negative log probability of the complete target word, summed over its
        GPT-2 subword tokens.
    entropy : float
        Entropy of the next-token probability distribution immediately before
        the target word.
    """
    if not context_ids:
        raise ValueError("The context must contain at least one GPT-2 token.")

    # Keep only the most recent context if GPT-2's context limit is exceeded.
    context_ids = context_ids[-MAX_CONTEXT_TOKENS:]

    context_tensor = torch.tensor(
        [context_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    # Distribution over the first subword token of the target word.
    context_logits = model(input_ids=context_tensor).logits[:, -1, :]
    context_probabilities = torch.softmax(context_logits, dim=-1)
    context_log_probabilities = torch.log_softmax(context_logits, dim=-1)

    entropy = float(
        (
            -context_probabilities * context_log_probabilities
        ).sum().item()
    )

    # A leading space tells GPT-2 that this is the next word after the context.
    target_ids = tokenizer.encode(
        " " + target_word,
        add_special_tokens=False,
    )

    if not target_ids:
        raise ValueError(
            f"The target word {target_word!r} produced no GPT-2 token IDs."
        )

    total_log_probability = 0.0
    running_ids = context_ids.copy()

    # A word may consist of multiple GPT-2 subword tokens. Calculate the
    # probability of each subword conditional on the previous subwords.
    for target_id in target_ids:
        current_ids = running_ids[-MAX_CONTEXT_TOKENS:]

        current_tensor = torch.tensor(
            [current_ids],
            dtype=torch.long,
            device=DEVICE,
        )

        logits = model(input_ids=current_tensor).logits[:, -1, :]
        log_probabilities = torch.log_softmax(logits, dim=-1)

        total_log_probability += float(
            log_probabilities[0, target_id].item()
        )

        running_ids.append(target_id)

    surprisal = -total_log_probability

    return surprisal, entropy


def calculate_from_words(context_words, target_word):
    """
    Convert a sequence of context words to GPT-2 IDs and calculate the scores.
    """
    if not context_words:
        raise ValueError("context_words cannot be empty.")

    context_text = " ".join(context_words)

    context_ids = tokenizer.encode(
        context_text,
        add_special_tokens=False,
    )

    return surprisal_entropy_from_ids(
        context_ids=context_ids,
        target_word=target_word,
    )


# ---------------------------------------------------------------------------
# Read and validate the input file
# ---------------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}\n\n"
        "Change INPUT_FILE in the configuration section."
    )

# sep=None lets pandas detect whether the file is comma-, semicolon-, or
# tab-separated. utf-8-sig also handles files saved with a UTF-8 BOM.
data = pd.read_csv(
    INPUT_FILE,
    sep=None,
    engine="python",
    encoding="utf-8-sig",
)

required_columns = {"code", "token", "text_nr", "item"}
missing_columns = required_columns.difference(data.columns)

if missing_columns:
    raise ValueError(
        "The input file is missing the following required columns: "
        f"{sorted(missing_columns)}.\n"
        f"Columns found: {list(data.columns)}"
    )

if data.empty:
    raise ValueError("The input file contains no rows.")

if data["text_nr"].isna().any():
    bad_rows = data.index[data["text_nr"].isna()].tolist()
    raise ValueError(
        f"Missing text_nr values in input rows: {bad_rows}"
    )

if data["token"].isna().any():
    bad_rows = data.index[data["token"].isna()].tolist()
    raise ValueError(
        f"Missing token values in input rows: {bad_rows}"
    )

# Convert tokens to strings while preserving punctuation and capitalization.
data["token"] = data["token"].astype(str)

empty_token_rows = data.index[data["token"].str.strip().eq("")].tolist()
if empty_token_rows:
    raise ValueError(
        f"Empty token values in input rows: {empty_token_rows}"
    )

# Preserve the original row order in the output.
data["_original_row_order"] = range(len(data))

# Use numeric item values for ordering when possible.
data["_item_numeric"] = pd.to_numeric(data["item"], errors="coerce")

# Result columns. Float NaN is used for the first two words of each text.
result_columns = [
    "surprisal_2_words_context",
    "entropy_2_words_context",
    "surprisal_full_context",
    "entropy_full_context",
]

for column in result_columns:
    data[column] = float("nan")


# ---------------------------------------------------------------------------
# Calculate scores separately for every text
# ---------------------------------------------------------------------------

number_of_texts = data["text_nr"].nunique(dropna=False)
number_of_rows = len(data)
processed_rows = 0

print(f"Found {number_of_texts} texts and {number_of_rows} rows.")

# sort=False keeps texts in the order in which they first occur in the file.
for text_counter, (text_nr, text_data) in enumerate(
    data.groupby("text_nr", sort=False),
    start=1,
):
    # Sort words within each text by item. Numeric ordering is used where
    # possible, and the original row order resolves ties.
    text_data = text_data.sort_values(
        by=["_item_numeric", "_original_row_order"],
        kind="stable",
        na_position="last",
    )

    row_indices = text_data.index.tolist()
    words = text_data["token"].tolist()

    print(
        f"\nText {text_counter}/{number_of_texts}: "
        f"text_nr={text_nr!r}, {len(words)} words"
    )

    if len(words) <= N_LOCAL_CONTEXT_WORDS:
        warnings.warn(
            f"Text {text_nr!r} contains only {len(words)} words. "
            "No word has two preceding context words."
        )
        continue

    for position in range(N_LOCAL_CONTEXT_WORDS, len(words)):
        dataframe_index = row_indices[position]
        target_word = words[position]

        # Exactly the two immediately preceding words.
        local_context_words = words[
            position - N_LOCAL_CONTEXT_WORDS:position
        ]

        # All preceding words from the beginning of the current text.
        full_context_words = words[:position]

        local_surprisal, local_entropy = calculate_from_words(
            context_words=local_context_words,
            target_word=target_word,
        )

        full_surprisal, full_entropy = calculate_from_words(
            context_words=full_context_words,
            target_word=target_word,
        )

        data.at[
            dataframe_index,
            "surprisal_2_words_context",
        ] = local_surprisal

        data.at[
            dataframe_index,
            "entropy_2_words_context",
        ] = local_entropy

        data.at[
            dataframe_index,
            "surprisal_full_context",
        ] = full_surprisal

        data.at[
            dataframe_index,
            "entropy_full_context",
        ] = full_entropy

        processed_rows += 1

        print(
            f"  item={text_data.loc[dataframe_index, 'item']!r}, "
            f"token={target_word!r} "
            f"({position + 1}/{len(words)})"
        )


# ---------------------------------------------------------------------------
# Restore original order and save results
# ---------------------------------------------------------------------------

data = data.sort_values(
    "_original_row_order",
    kind="stable",
)

data = data.drop(
    columns=[
        "_original_row_order",
        "_item_numeric",
    ]
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

data.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("\nFinished.")
print(f"Calculated scores for {processed_rows} rows.")
print(f"Output written to:\n{OUTPUT_FILE}")
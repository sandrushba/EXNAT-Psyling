README

Familiarity, valence, concreteness, and arousal:
1. python prepare_experiment.py [experiment_path] [experiment_name]
2. python execute_experiment.py [experiment_path] [experiment_name]
3. Convert output JSON file to CSV.
4. Run "get_rating.Rmd" 

Age-of-acquisition:
1. python prepare_experiment.py [experiment_path] [experiment_name]
2. python execute_experiment.py [experiment_path] [experiment_name]
3. python generateResults.py [mode] [language]

Fine-tuning an LLM:
1. python create_finetuning_dataset.py [experiment_path] [experiment_name]
2. python execute_finetune.py [experiment_path] [experiment_name]

Source of the Python scripts and the toolbox in general:
Conde, J., Martínez, G., Grandury, M., Arriaga, C., Haro, J., Schroeder, S., Hintz, F., Reviriego, P., & Brysbaert, M. (2026). Updating the German Psycholinguistic Word Toolbox with AI-Generated Estimates of Concreteness, Valence, Arousal, Age of Acquisition, and Familiarity. Journal of Cognition, 9(1), 9. https://doi.org/10.5334/joc.482

Source of the Glasgow human norms data:
Scott, G. G., Keitel, A., Becirspahic, M., Yao, B., & Sereno, S. C. (2019). The Glasgow Norms: Ratings of 5,500 words on nine scales. Behavior Research Methods, 51(3), 1258–1270. https://doi.org/10.3758/s13428-018-1099-3

Source of the German age-of-acquisition model training data:
Birchenough, J. M. H., Davies, R., & Connelly, V. (2017). Rated age-of-acquisition norms for over 3,200 German words. Behavior Research Methods, 49(2), 484–501. https://doi.org/10.3758/s13428-016-0718-0

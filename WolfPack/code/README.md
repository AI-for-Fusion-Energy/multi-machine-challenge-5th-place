### Within best_models:
- Contains catboost and xgboost pickled models trained on each fold

### Within processed_data:

- FINAL_MMD_JT_processed_sum_feature_extraction_train.csv:  Preprocessed JTEXT train dataframe.

- FINAL_MMD_HL_processed_sum_feature_extraction_train.csv:  Preprocessed HL_2A train dataframe

- FINAL_MMD_CM_processed_sum_feature_extraction_train.csv:  Preprocessed C-MOD train dataframe

- FINAL_MMD_CM_processed_sum_feature_extraction_test.csv:  Preprocessed C-MOD test dataframe

-FINAL_MMD_JT_HL_CM_MMD_PIPELINE1_1.csv:  saved  inference file

## Others

- FINAL_MMD_JT_HL_CM.ipynb: Preprocessing and Modelling Notebook

- utils.py: scripts for preprocessing jtext, cmod and HL_2A shot datasets into dataframes for the modelling pipeline

- basic_processor.py: basic processor provided to participants by ITU

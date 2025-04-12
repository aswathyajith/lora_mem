#!/bin/bash

config_df="config_dfs/configurations.csv"

stdout_logs="$STDOUT_PATH/logs-$DOMAIN.out"

echo "python src/preprocess/process_datasets.py --config_df $config_df --path_to_save_freqs "data/processed/tkn_freqs" --domain $DOMAIN >> $stdout_logs 2>&1"

python src/preprocess/process_dataset.py --config_df $config_df --path_to_save_freqs "data/processed/tkn_freqs" --domain $DOMAIN >> $stdout_logs 2>&1

echo "Preprocessing done"
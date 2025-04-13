#!/bin/bash

config_df="config_dfs/configurations.csv"

stdout_logs="$STDOUT_PATH/logs-$DOMAIN.out"

echo "python src/training/select_models.py --domain $DOMAIN >> $stdout_logs 2>&1"

python src/training/select_models.py --domain $DOMAIN >> $stdout_logs 2>&1

echo "Preprocessing done"
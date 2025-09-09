#!/bin/bash

DATA_FOLDER="data"

AGENTS=("codofuzz")

# Define allowed query sizes for each dataset as per the table
declare -A QUERY_SIZES
QUERY_SIZES["splice"]="1 5 20 50 100"
# # QUERY_SIZES["dna"]="1 5 20 50 100"
# # QUERY_SIZES["usps"]="1 5 20 50 100"
# # QUERY_SIZES["cifar10"]="500 1000"
# QUERY_SIZES["fashionmnist"]="500 1000"
# QUERY_SIZES["mnist"]="1 5 20 50 100"         # "Semi FMnist" row
# QUERY_SIZES["topv2"]="1 5 20 50"
# QUERY_SIZES["news"]="20 50 100 500"

for agent in "${AGENTS[@]}"; do
    for dataset in "${!QUERY_SIZES[@]}"; do
        for query_size in ${QUERY_SIZES[$dataset]}; do
            echo "Running evaluation for agent: $agent on dataset: $dataset with query size: $query_size"
            python evaluate.py --data_folder "$DATA_FOLDER" --agent "$agent" --dataset "$dataset" --query_size "$query_size" --restarts 5
        done
    done
done
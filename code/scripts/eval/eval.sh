#!/bin/bash

# Ensure a graph name is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <CSV_FILE> <AUG_FILE> <DATASET_NAME> <OUTPUT_PATH>"
    exit 1
fi

CSV_FILE="$1"
AUG_FILE="$2"
DATASET_NAME="$3"
OUTPUT_PATH="$4"

# Read the header line and convert it into an array of parameter names
IFS=',' read -r -a headers < "$CSV_FILE"
IFS=',' read -r -a aug_headers < "$AUG_FILE"

# Find the index of the "graph" column

dataset_index=-1
epochs_index=-1
model_index=-1
for i in "${!headers[@]}"; do
    if [[ "${headers[i]}" == "dataset" ]]; then
        dataset_index=$i
    elif [[ "${headers[i]}" == "base_model" ]]; then
        model_index=$i
    elif [[ "${headers[i]}" == "epochs" ]]; then
        epochs_index=$i
    fi
done

# If "graph" column is not found, exit with an error
if [[ $dataset_index -eq -1 ]]; then
    echo "Error: CSV file must contain a 'dataset' column for the dataset name."
    exit 1
fi

# Expected parameters in CSV file:
# dataset
# base_model
# lr
# batch size
# epochs

# Expected parameters in AUG file:
# n_negative
# lam

shared_args="--test_set=test seed=2020 recdim=128 board_path=$OUTPUT_PATH"

# Read the rest of the CSV file, skipping the header
tail -n +2 "$CSV_FILE" | while IFS=',' read -r "${headers[@]}"; do
    # Check if this row's graph name matches the specified graph name
    if [[ "${!headers[dataset_index]}" == "$DATASET_NAME" ]]; then
        model_name="${!headers[model_index]}"

        # Construct named arguments dynamically
        args=()
        for i in "${!headers[@]}"; do
            args+=("--${headers[i]}=${!headers[i]}")
        done

        # echo "$shared_args ${args[@]}"
        # vanilla
        # python main.py --loss_func=sg "$shared_args ${args[@]} --board_path=$OUTPUT_PATH" 

        # weighted vanilla
        # python main.py --loss_func=sg "$shared_args ${args[@]}" --n_negative=-1 --K=5 --alpha=0.75 "--board_path=$OUTPUT_PATH" 

        if [[ $epochs_index -ne -1 ]]; then
            args[$epochs_index]="--epochs=2"
        else
            echo "Warning: 'epochs' column not found. Running second execution without modification."
        fi
        # echo "$shared_args ${args[@]}"

        # no negative
        # python main.py --loss_func=sg_aug "$shared_args ${args[@]}" --n_negative=1000000000 "--board_path=$OUTPUT_PATH"
        args[$epochs_index]="--epochs=${!headers[$epochs_index]}"

        tail -n +2 "$AUG_FILE" | while IFS=',' read -r "${aug_headers[@]}"; do
            if [[ "${!aug_headers[dataset_index]}" == "$DATASET_NAME" && "${!aug_headers[model_index]}" == "$model_name" ]]; then
                aug_args=()
                for i in "${!aug_headers[@]}"; do
                    if [[ $i -ne $dataset_index ]]; then
                        aug_args+=("--${aug_headers[i]}=${!aug_headers[i]}")
                    fi
                done
                # echo "${aug_args[@]}"
                echo "python main.py --loss_func=sg_aug $shared_args ${args[@]} ${aug_args[@]} --board_path=$OUTPUT_PATH"
                # python main.py --loss_func=sg_aug "$shared_args ${args[@]} ${aug_args[@]} --board_path=$OUTPUT_PATH"
            fi
        done
    fi
done

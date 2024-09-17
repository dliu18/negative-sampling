#!/bin/bash

# Define the values for base_model and loss_func
dataset="Cora"
base_models=("n2v" "line")
loss_funcs=("sg" "sg_aug")
lrs=(1.0 0.1 0.01)
n_negatives=(1 2 4 8)

# Iterate over all combinations of base_model and loss_func
for base_model in "${base_models[@]}"; do
    for loss_func in "${loss_funcs[@]}"; do
        # Set batch_size based on the base_model
        if [ "$base_model" == "n2v" ]; then
            batch_size=128
        elif [ "$base_model" == "line" ]; then
            batch_size=1024
        else
            echo "Unknown base_model: $base_model"
            continue
        fi

        if [ "$loss_func" == "sg" ]; then
            for lr in "${lrs[@]}"; do
                # Execute the command with the current combination of base_model and loss_func
                echo "Model: $base_model \t Loss: $loss_func \t LR: $lr \t n_negative: 1"
                python main.py \
                --base_model="$base_model" \
                --loss_func="$loss_func" \
                --n_negative=1 \
                --lr="$lr" \
                --seed=2020 \
                --dataset="$dataset" \
                --recdim=128 \
                --batch_size="$batch_size" \
                --epochs 3
            done
        else
            for n_negative in "${n_negatives[@]}"; do
                for lr in "${lrs[@]}"; do
                    # Execute the command with the current combination of base_model and loss_func
                    echo "Model: $base_model \t Loss: $loss_func \t LR: $lr \t n_negative: $n_negative"
                    python main.py \
                    --base_model="$base_model" \
                    --loss_func="$loss_func" \
                    --n_negative="$n_negative" \
                    --lr="$lr" \
                    --seed=2020 \
                    --dataset="$dataset" \
                    --recdim=128 \
                    --batch_size="$batch_size" \
                    --epochs 3
                done
            done
        fi
    done
done

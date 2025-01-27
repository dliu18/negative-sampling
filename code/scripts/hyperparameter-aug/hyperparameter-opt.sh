#!/bin/bash

# Define the values for base_model and loss_func
#./hyperparameter-opt.sh dataset epochs batch_size_line name
dataset="$1"
epochs="$2"
lr="$3"

# base_models=("n2v" "line")
base_models=("n2v" "line")
loss_funcs=("sg" "sg_aug")

# n_negatives=(1 4 10 1000000000)
n_negatives=(1 10 100 1000000000)
lams=(100.0 10.0 1.0 0.1)
# lams=(1.0 0.1 0.01)

# Iterate over all combinations of base_model and loss_func
for base_model in "${base_models[@]}"; do
    for loss_func in "${loss_funcs[@]}"; do
        # Set batch_size based on the base_model
        if [ "$base_model" == "n2v" ]; then
            batch_size="$4"
        elif [ "$base_model" == "line" ]; then
            batch_size="$5"
        else
            echo "Unknown base_model: $base_model"
            continue
        fi

        for n_negative in "${n_negatives[@]}"; do
            for lam in "${lams[@]}"; do
                # Execute the command with the current combination of base_model and loss_func
                echo "Model: $base_model \t Loss: $loss_func \t LR: $lr lam: $lam \t n_negative: $n_negative"
                python main.py \
                --base_model="$base_model" \
                --loss_func="$loss_func" \
                --test_set="valid" \
                --n_negative="$n_negative" \
                --lr="$lr" \
                --lam="$lam" \
                --seed=2020 \
                --dataset="$dataset" \
                --recdim=128 \
                --batch_size="$batch_size" \
                --epochs="$epochs"\
                --board_path="$6"
            done
        done
        fi
    done
done

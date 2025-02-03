#!/bin/bash

# Define the values for base_model and loss_func
#./hyperparameter-opt.sh dataset epochs classifier_epochs lr batch_size_line batch_size_n2v n2v_p n2v_q name
dataset="$1"
epochs="$2"
classifier_epochs="$3"
batch_size="$4"
n2v_p="$7"
n2v_q="$8"
board_path="$9"

# base_models=("n2v" "line")
base_models=("line")
loss_funcs=("sg_aug")

# n_negatives=(1 5 10 100)
n_negatives=(100)
lams=(100.0 10.0 1.0 0.1)

# Iterate over all combinations of base_model and loss_func
for base_model in "${base_models[@]}"; do
    for loss_func in "${loss_funcs[@]}"; do
        # Set batch_size based on the base_model
        if [ "$base_model" == "n2v" ]; then
            lr="$5"
        elif [ "$base_model" == "line" ]; then
            lr="$6"
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
                --n2v_p="$n2v_p"\
                --n2v_q="$n2v_q"\
                --lam="$lam" \
                --seed=2020 \
                --dataset="$dataset" \
                --recdim=128 \
                --batch_size="$batch_size" \
                --epochs="$epochs"\
                --classifier_epochs="$classifier_epochs"\
                --board_path="$board_path"
            done
        done
    done
done

#!/bin/bash

# Define the values for base_model and loss_func
#./hyperparameter-opt.sh dataset epochs batch_size_line name

graphs=("Cora" "CiteSeer" "PubMed")
base_models=("n2v" "line")
test_set_fracs=(0.85 0.65 0.45 0.25)
n_negatives=(10 100 1000000000)
lams=(0.1 1 10 100)

# graphs=("Cora" "CiteSeer" "PubMed")
# base_models=("n2v" "line")
# test_set_fracs=(0.25)
# n_negatives=(10)
# lams=(10)

for dataset in "${graphs[@]}"; do
    for base_model in "${base_models[@]}"; do
        for test_set_frac in "${test_set_fracs[@]}"; do
            python main.py \
            --base_model="$base_model" \
            --loss_func="sg" \
            --test_set="test" \
            --test_set_frac="$test_set_frac" \
            --lr=0.1 \
            --seed=2020 \
            --dataset="$dataset" \
            --recdim=128 \
            --batch_size=128 \
            --epochs=20\
            --board_path="density-jan-2"

            for n_negative in "${n_negatives[@]}"; do
                for lam in "${lams[@]}"; do
                    python main.py \
                    --base_model="$base_model" \
                    --loss_func="sg_aug" \
                    --test_set="test" \
                    --test_set_frac="$test_set_frac" \
                    --lam="$lam" \
                    --n_negative="$n_negative" \
                    --lr=0.1 \
                    --seed=2020 \
                    --dataset="$dataset" \
                    --recdim=128 \
                    --batch_size=128 \
                    --epochs=20\
                    --board_path="density-jan-2"
                done        
            done 
        done
    done
done
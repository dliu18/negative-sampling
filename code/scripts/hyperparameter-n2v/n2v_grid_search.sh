#!/bin/bash

dataset="$1"
lr="$2"
batch_size="$3"
epochs="$4"
path="$5"

ps=(0.25 0.50 1.0 2.0 4.0)
qs=(0.25 0.50 1.0 2.0 4.0)

for p in "${ps[@]}"; do
    for q in "${qs[@]}"; do
		python main.py \
		--base_model="n2v" \
		--loss_func="sg" \
		--test_set="test" \
		--lr="$lr" \
		--n2v_p="$p" \
		--n2v_q="$q" \
		--seed=2020 \
		--dataset="$dataset" \
		--recdim=128 \
		--batch_size="$batch_size" \
		--epochs="$epochs" \
		--board_path="$path"
    done
done

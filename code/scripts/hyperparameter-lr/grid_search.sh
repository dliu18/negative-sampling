#!/bin/bash

dataset="$1"
base_model="$2"
batch_size="$3"
epochs="$4"
path="$5"

lrs=(0.001 0.01 0.1 1)

for lr in "${lrs[@]}"; do
	python main.py \
	--base_model="$base_model" \
	--loss_func="sg" \
	--test_set="test" \
	--lr="$lr" \
	--seed=2020 \
	--dataset="$dataset" \
	--recdim=128 \
	--batch_size="$batch_size" \
	--epochs="$epochs" \
	--board_path="$path"
done
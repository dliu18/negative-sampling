#!/bin/bash

ps=(0.95 0.8 0.6 0.4 0.2 0.2 0.2 0.2 0.2 0.2 0.2)
qs=(0.004 0.004 0.004 0.004 0.004 0.02 0.04 0.08 0.12 0.16 0.2)

# Get the length of the first list (assuming both lists have the same length)
length=${#ps[@]}

# Loop through the indices of the arrays
for (( i=0; i<$length; i++ ))
do
	# python main.py \
	# --base_model="line" \
	# --loss_func="sg" \
	# --test_set="test" \
	# --lr=0.1 \
	# --seed=2020 \
	# --dataset="SBM-${ps[$i]}-${qs[$i]}" \
	# --recdim=128 \
	# --batch_size=256 \
	# --epochs=5 \
	# --board_path="sbm-line-extended"

	# python main.py \
	# --base_model="line" \
	# --loss_func="sg_aug" \
	# --test_set="test" \
	# --n_negative=10 \
	# --lr=0.1 \
	# --lam=0.1 \
	# --seed=2020 \
	# --dataset="SBM-${ps[$i]}-${qs[$i]}" \
	# --recdim=128 \
	# --batch_size=256 \
	# --epochs=5 \
	# --board_path="sbm-line-extended"

	python main.py \
	--base_model="line" \
	--loss_func="sg_aug" \
	--test_set="test" \
	--n_negative=1000000000 \
	--lr=0.1 \
	--lam=0.1 \
	--seed=2020 \
	--dataset="SBM-${ps[$i]}-${qs[$i]}" \
	--recdim=128 \
	--batch_size=256 \
	--epochs=5 \
	--board_path="sbm-line-extended"
done
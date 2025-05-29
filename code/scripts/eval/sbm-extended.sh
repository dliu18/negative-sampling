#!/bin/bash

path="$1"

# rates from ICLR submission
# ps=(0.95 0.8 0.6 0.4 0.2 0.2 0.2 0.2 0.2 0.2 0.2)
# qs=(0.004 0.004 0.004 0.004 0.004 0.02 0.04 0.08 0.12 0.16 0.2)

# 1/10 of original rates
# ps=(0.095 0.08 0.06 0.04 0.02 0.02 0.02 0.02 0.02 0.02 0.02)
# qs=(0.0004 0.0004 0.0004 0.0004 0.0004 0.002 0.004 0.008 0.012 0.016 0.02)

# 1/2 of original rates [KEEP]
# ps=(0.475 0.4 0.3 0.2 0.1 0.1 0.1 0.1 0.1 0.1 0.1)
# qs=(0.002 0.002 0.002 0.002 0.002 0.01 0.02 0.04 0.06 0.08 0.1)

ps=(0.49 0.48 0.47 0.46 0.45 0.44 0.43 0.42 0.41)
qs=(0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002)
# density experiments
# ps=(0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2)
# qs=(0.002 0.004 0.006 0.008 0.01 0.012 0.014 0.016 0.018 0.02)

#ICLR
# ps=(0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1)
# qs=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)


# Get the length of the first list (assuming both lists have the same length)
length=${#ps[@]}

# Loop through the indices of the arrays
for (( i=0; i<$length; i++ ))
do
	# python main.py \
	# --base_model="n2v" \
	# --loss_func="sg" \
	# --test_set="test" \
	# --lr=0.01 \
	# --n2v_q=2 \
	# --seed=2020 \
	# --dataset="SBM-${ps[$i]}-${qs[$i]}" \
	# --recdim=128 \
	# --batch_size=128 \
	# --epochs=5 \
	# --classifier_epochs=3 \
	# --board_path="$path"

	# python main.py \
	# --base_model="n2v" \
	# --loss_func="sg_aug" \
	# --test_set="test" \
	# --n_negative=100 \
	# --lr=0.01 \
	# --n2v_q=2 \
	# --lam=100 \
	# --seed=2020 \
	# --dataset="SBM-${ps[$i]}-${qs[$i]}" \
	# --recdim=128 \
	# --batch_size=128 \
	# --epochs=5 \
	# --classifier_epochs=3 \
	# --board_path="$path"

	# python main.py \
	# --base_model="n2v" \
	# --loss_func="sg_aug" \
	# --test_set="test" \
	# --n_negative=10 \
	# --lr=0.01 \
	# --n2v_q=2 \
	# --lam=100 \
	# --seed=2020 \
	# --dataset="SBM-${ps[$i]}-${qs[$i]}" \
	# --recdim=128 \
	# --batch_size=128 \
	# --epochs=5 \
	# --classifier_epochs=3 \
	# --board_path="$path"

	# python main.py \
	# --base_model="n2v" \
	# --loss_func="sg_aug" \
	# --test_set="test" \
	# --n_negative=1000000000 \
	# --lr=0.01 \
	# --n2v_q=2 \
	# --lam=0.1 \
	# --seed=2020 \
	# --dataset="SBM-${ps[$i]}-${qs[$i]}" \
	# --recdim=128 \
	# --batch_size=128 \
	# --epochs=5 \
	# --classifier_epochs=3 \
	# --board_path="$path"

	# python main.py \
	# --base_model="line" \
	# --loss_func="sg" \
	# --test_set="test" \
	# --lr=0.01 \
	# --seed=2020 \
	# --dataset="SBM-${ps[$i]}-${qs[$i]}" \
	# --recdim=128 \
	# --batch_size=128 \
	# --epochs=5 \
	# --classifier_epochs=3 \
	# --board_path="$path"

	# python main.py \
	# --base_model="line" \
	# --loss_func="sg_aug" \
	# --test_set="test" \
	# --n_negative=10 \
	# --lr=0.01 \
	# --lam=1 \
	# --seed=2020 \
	# --dataset="SBM-${ps[$i]}-${qs[$i]}" \
	# --recdim=128 \
	# --batch_size=128 \
	# --epochs=5 \
	# --classifier_epochs=3 \
	# --board_path="$path"

	# python main.py \
	# --base_model="line" \
	# --loss_func="sg_aug" \
	# --test_set="test" \
	# --n_negative=100 \
	# --lr=0.01 \
	# --lam=1 \
	# --seed=2020 \
	# --dataset="SBM-${ps[$i]}-${qs[$i]}" \
	# --recdim=128 \
	# --batch_size=128 \
	# --epochs=5 \
	# --classifier_epochs=3 \
	# --board_path="$path"

	python main.py \
	--base_model="line" \
	--loss_func="sg_aug" \
	--test_set="test" \
	--n_negative=1000000000 \
	--lr=0.01 \
	--lam=0.1 \
	--seed=9999 \
	--dataset="SBM-${ps[$i]}-${qs[$i]}" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=5 \
	--classifier_epochs=3 \
	--board_path="$path"
done
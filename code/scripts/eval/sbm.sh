#!/bin/bash

q_s=(0.004 0.008 0.012 0.016 0.02 0.024 0.028 0.032 0.036 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2)

# q_s=(0.0005 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.015 0.02 0.012 0.03 0.035 0.04 0.045 0.05)
# q_s=(0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2)


for q in "${q_s[@]}"; do
	python main.py \
	--base_model="line" \
	--loss_func="sg" \
	--test_set="test" \
	--lr=0.1 \
	--seed=2020 \
	--dataset="SBM-0.2-$q" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=10 \
	--board_path="sbm-line-centered"

	python main.py \
	--base_model="line" \
	--loss_func="sg" \
	--test_set="test" \
	--n_negative=-1 \
	--lr=0.1 \
	--K=5 \
	--alpha=0.75 \
	--seed=2020 \
	--dataset="SBM-0.2-$q" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=10 \
	--board_path="sbm-line-centered"

	python main.py \
	--base_model="line" \
	--loss_func="sg_aug" \
	--test_set="test" \
	--n_negative=10 \
	--lr=0.1 \
	--lam=0.1 \
	--seed=2020 \
	--dataset="SBM-0.2-$q" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=10 \
	--board_path="sbm-line-centered"

	python main.py \
	--base_model="line" \
	--loss_func="sg_aug" \
	--test_set="test" \
	--n_negative=10 \
	--lr=0.1 \
	--lam=0.1 \
	--alpha=0.75 \
	--seed=2020 \
	--dataset="SBM-0.2-$q" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=10 \
	--board_path="sbm-weighted-line-centered"

	python main.py \
	--base_model="line" \
	--loss_func="sg_aug" \
	--test_set="test" \
	--n_negative=1000000000 \
	--lr=0.1 \
	--lam=0.1 \
	--seed=2020 \
	--dataset="SBM-0.2-$q" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=10 \
	--board_path="sbm-line-centered"
done


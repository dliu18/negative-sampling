#!/bin/bash

q_s=(0.004 0.008 0.012 0.016 0.02 0.024 0.028 0.032 0.036 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2)

for q in "${q_s[@]}"; do
	python main.py \
	--base_model="n2v" \
	--loss_func="sg" \
	--test_set="test" \
	--lr=0.1 \
	--seed=2020 \
	--dataset="SBM-0.2-$q" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=25 \
	--board_path="sbm"

	python main.py \
	--base_model="n2v" \
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
	--epochs=25 \
	--board_path="sbm"

	python main.py \
	--base_model="n2v" \
	--loss_func="sg_aug" \
	--test_set="test" \
	--n_negative=10 \
	--lr=0.1 \
	--lam=0.1 \
	--seed=2020 \
	--dataset="SBM-0.2-$q" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=25 \
	--board_path="sbm"

	python main.py \
	--base_model="n2v" \
	--loss_func="sg_aug" \
	--test_set="test" \
	--n_negative=1000000 \
	--lr=0.1 \
	--lam=0.1 \
	--seed=2020 \
	--dataset="SBM-0.2-$q" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=25 \
	--board_path="sbm"
done


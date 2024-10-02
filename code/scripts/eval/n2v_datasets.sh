#!/bin/bash

datasets=(ego-facebook soc-ca-astroph)

for dataset in "${datasets[@]}"; do
	python main.py \
	--base_model="n2v" \
	--loss_func="sg" \
	--test_set="test" \
	--lr=0.1 \
	--seed=2020 \
	--dataset="$dataset"\
	--recdim=128 \
	--batch_size=128 \
	--epochs=25 \
	--board_path="n2v_datasets"

	python main.py \
	--base_model="n2v" \
	--loss_func="sg" \
	--test_set="test" \
	--n_negative=-1 \
	--lr=0.1 \
	--K=5 \
	--alpha=0.75 \
	--seed=2020 \
	--dataset="$dataset" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=25 \
	--board_path="n2v_datasets"

	python main.py \
	--base_model="n2v" \
	--loss_func="sg_aug" \
	--test_set="test" \
	--n_negative=10 \
	--lr=0.1 \
	--lam=0.1 \
	--seed=2020 \
	--dataset="$dataset" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=25 \
	--board_path="n2v_datasets"

	python main.py \
	--base_model="n2v" \
	--loss_func="sg_aug" \
	--test_set="test" \
	--n_negative=10 \
	--lr=0.1 \
	--lam=0.1 \
	--alpha=0.75 \
	--seed=2020 \
	--dataset="$dataset" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=25 \
	--board_path="n2v_datasets_weighted"

	python main.py \
	--base_model="n2v" \
	--loss_func="sg_aug" \
	--test_set="test" \
	--n_negative=1000000000 \
	--lr=0.1 \
	--lam=0.1 \
	--seed=2020 \
	--dataset="$dataset" \
	--recdim=128 \
	--batch_size=128 \
	--epochs=25 \
	--board_path="n2v_datasets"
done


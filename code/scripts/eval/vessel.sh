#!/bin/bash

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.01 \
--n2v_p=0.5 \
--n2v_q=0.25 \
--seed=2020 \
--dataset="ogbl-vessel" \
--recdim=128 \
--batch_size=2048 \
--epochs=3 \
--board_path="jan-29"

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.01 \
--n2v_p=0.5 \
--n2v_q=0.25 \
--n_negative=-1 \
--K=5 \
--alpha=0.75 \
--seed=2020 \
--dataset="ogbl-vessel" \
--recdim=128 \
--batch_size=2048 \
--epochs=3 \
--board_path="jan-29"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=5 \
--lr=0.01 \
--lam=1 \
--n2v_p=0.5 \
--n2v_q=0.25 \
--seed=2020 \
--dataset="ogbl-vessel" \
--recdim=128 \
--batch_size=2048 \
--epochs=3 \
--board_path="jan-29"


python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.01 \
--lam=0.000001 \
--n2v_p=0.5 \
--n2v_q=0.25 \
--seed=2020 \
--dataset="ogbl-vessel" \
--recdim=128 \
--batch_size=2048 \
--epochs=2 \
--board_path="jan-29"

python main.py \
--base_model="line" \
--loss_func="sg" \
--test_set="test" \
--lr=0.01 \
--seed=2020 \
--dataset="ogbl-vessel" \
--recdim=128 \
--batch_size=2048 \
--epochs=3 \
--board_path="jan-29"

python main.py \
--base_model="line" \
--loss_func="sg" \
--test_set="test" \
--lr=0.01 \
--n_negative=-1 \
--K=5 \
--alpha=0.75 \
--seed=2020 \
--dataset="ogbl-vessel" \
--recdim=128 \
--batch_size=2048 \
--epochs=3 \
--board_path="jan-29"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=100 \
--lr=0.01 \
--lam=1 \
--seed=2020 \
--dataset="ogbl-vessel" \
--recdim=128 \
--batch_size=2048 \
--epochs=3 \
--board_path="jan-29"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.01 \
--seed=2020 \
--dataset="ogbl-vessel" \
--recdim=128 \
--batch_size=2048 \
--epochs=2 \
--board_path="jan-29"
#!/bin/bash

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.01 \
--n2v_p=2 \
--n2v_q=4 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=2048 \
--epochs=5 \
--board_path="jan-29"

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.01 \
--n2v_p=2 \
--n2v_q=4 \
--n_negative=-1 \
--K=5 \
--alpha=0.75 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=2048 \
--epochs=5 \
--board_path="jan-29"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=10 \
--lr=0.01 \
--n2v_p=2 \
--n2v_q=4 \
--lam=100 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=2048 \
--epochs=5 \
--board_path="jan-29"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.01 \
--n2v_p=2 \
--n2v_q=4 \
--seed=2020 \
--dataset="ogbl-ppa" \
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
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=2048 \
--epochs=5 \
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
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=2048 \
--epochs=5 \
--board_path="jan-29"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=5 \
--lr=0.01 \
--lam=100 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=2048 \
--epochs=5 \
--board_path="jan-29"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.01 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=2048 \
--epochs=2 \
--board_path="jan-29"
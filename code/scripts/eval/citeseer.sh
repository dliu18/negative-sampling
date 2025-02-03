#!/bin/bash

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.1 \
--n2v_p=1 \
--n2v_q=0.5 \
--seed=2020 \
--dataset="CiteSeer" \
--recdim=128 \
--batch_size=128 \
--epochs=20 \
--board_path="jan-29"

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--n_negative=-1 \
--alpha=0.75 \
--K=5 \
--lr=0.1 \
--n2v_p=1 \
--n2v_q=0.5 \
--seed=2020 \
--dataset="CiteSeer" \
--recdim=128 \
--batch_size=128 \
--epochs=20 \
--board_path="jan-29"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=100 \
--lr=0.1 \
--lam=100 \
--n2v_p=1 \
--n2v_q=0.5 \
--seed=2020 \
--dataset="CiteSeer" \
--recdim=128 \
--batch_size=128 \
--epochs=20 \
--board_path="jan-29"


python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.1 \
--n2v_p=1 \
--n2v_q=0.5 \
--seed=2020 \
--dataset="CiteSeer" \
--recdim=128 \
--batch_size=128 \
--epochs=2 \
--board_path="jan-29"

python main.py \
--base_model="line" \
--loss_func="sg" \
--test_set="test" \
--lr=0.1 \
--seed=2020 \
--dataset="CiteSeer" \
--recdim=128 \
--batch_size=128 \
--epochs=20 \
--board_path="jan-29"

python main.py \
--base_model="line" \
--loss_func="sg" \
--test_set="test" \
--lr=0.1 \
--seed=2020 \
--dataset="CiteSeer" \
--recdim=128 \
--batch_size=128 \
--epochs=20 \
--n_negative=-1 \
--alpha=0.75 \
--K=5 \
--board_path="jan-29"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=100 \
--lr=0.1 \
--lam=0.1 \
--seed=2020 \
--dataset="CiteSeer" \
--recdim=128 \
--batch_size=128 \
--epochs=20 \
--board_path="jan-29"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.1 \
--seed=2020 \
--dataset="CiteSeer" \
--recdim=128 \
--batch_size=128 \
--epochs=2 \
--board_path="jan-29"
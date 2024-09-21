#!/bin/bash

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.01 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=1024 \
--epochs=2

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=10 \
--lr=0.01 \
--lam=0.001 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=1024 \
--epochs=2

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=20 \
--lr=0.01 \
--lam=0.001 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=1024 \
--epochs=2

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=50 \
--lr=0.01 \
--lam=0.001 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=1024 \
--epochs=2

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=100 \
--lr=0.01 \
--lam=0.001 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=1024 \
--epochs=2

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=2 \
--lr=0.01 \
--lam=0.001 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=1024 \
--epochs=2

# python main.py \
# --base_model="line" \
# --loss_func="sg" \
# --test_set="test" \
# --lr=0.01 \
# --seed=2020 \
# --dataset="ogbl-ppa" \
# --recdim=128 \
# --batch_size=64 \
# --epochs=2

# python main.py \
# --base_model="line" \
# --loss_func="sg_aug" \
# --test_set="test" \
# --n_negative=2 \
# --lr=0.01 \
# --lam=0.001 \
# --seed=2020 \
# --dataset="ogbl-ppa" \
# --recdim=128 \
# --batch_size=64 \
# --epochs=2

# python main.py \
# --base_model="line" \
# --loss_func="sg_aug" \
# --test_set="test" \
# --n_negative=1000 \
# --lr=0.01 \
# --lam=0.001 \
# --seed=2020 \
# --dataset="ogbl-ppa" \
# --recdim=128 \
# --batch_size=64 \
# --epochs=2
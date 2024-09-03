#!/bin/bash

python main.py --decay=1e-4 --lr=0.001 --layer=0 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="l2" --reg_lam 0.0  --epochs 100 --testbatch 50

python main.py --decay=1e-4 --lr=0.001 --layer=0 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="l2" --reg_lam 0.5 --epochs 100 --testbatch 50
python main.py --decay=1e-4 --lr=0.001 --layer=1 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="l2" --reg_lam 0.5 --epochs 100 --testbatch 50
python main.py --decay=1e-4 --lr=0.001 --layer=2 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="l2" --reg_lam 0.5 --epochs 100 --testbatch 50
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="l2" --reg_lam 0.5 --epochs 100 --testbatch 50
python main.py --decay=1e-4 --lr=0.001 --layer=4 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="l2" --reg_lam 0.5 --epochs 100 --testbatch 50
python main.py --decay=1e-4 --lr=0.001 --layer=5 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="l2" --reg_lam 0.5 --epochs 100 --testbatch 50

python main.py --decay=1e-4 --lr=0.001 --layer=0 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="bpr" --reg_lam 0.0 --epochs 100 --testbatch 50

python main.py --decay=1e-4 --lr=0.001 --layer=0 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="bpr" --reg_lam 0.5 --epochs 100 --testbatch 50
python main.py --decay=1e-4 --lr=0.001 --layer=1 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="bpr" --reg_lam 0.5 --epochs 100 --testbatch 50
python main.py --decay=1e-4 --lr=0.001 --layer=2 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="bpr" --reg_lam 0.5 --epochs 100 --testbatch 50
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="bpr" --reg_lam 0.5 --epochs 100 --testbatch 50
python main.py --decay=1e-4 --lr=0.001 --layer=4 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="bpr" --reg_lam 0.5 --epochs 100 --testbatch 50
python main.py --decay=1e-4 --lr=0.001 --layer=5 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim=10 --loss_func="bpr" --reg_lam 0.5 --epochs 100 --testbatch 50


#!/bin/bash

# Initial values for the arguments
args=(2 5 10)

# Add remaining values until 300
for ((i=20; i<=300; i+=10)); do
    args+=($i)
done

# Loop through each argument and call the Python script
for arg in "${args[@]}"; do
    python main.py --decay=1e-4 --lr=0.001 --layer=1 --seed=2020 --dataset="lastfm-small" --topks="[20]" --recdim="$arg"
done


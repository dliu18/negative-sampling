#!/bin/bash

python post-process.py \
--seed=2020 \
--recdim=128 \
--base_model="$1" \
--test_set="test" \
--load=1 \
--board_path="kdd-25"
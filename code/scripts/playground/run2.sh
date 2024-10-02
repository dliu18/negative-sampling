python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000 \
--lr=0.01 \
--lam=0.0001 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=256 \
--epochs=3 \
--board_path="scratch"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000 \
--lr=0.00001 \
--lam=0.01 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=256 \
--epochs=3 \
--board_path="scratch"
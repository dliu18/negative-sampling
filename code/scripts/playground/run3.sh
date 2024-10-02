python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000 \
--lr=0.01 \
--lam=0.000001 \
--seed=2020 \
--dataset="ogbl-citation2" \
--recdim=128 \
--batch_size=1024 \
--epochs=3 \
--board_path="scratch"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000 \
--lr=0.01 \
--lam=0.000001 \
--seed=2020 \
--dataset="ogbl-citation2" \
--recdim=128 \
--batch_size=1024 \
--epochs=2 \
--board_path="scratch"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000 \
--lr=0.01 \
--lam=0.0000001 \
--seed=2020 \
--dataset="ogbl-citation2" \
--recdim=128 \
--batch_size=1024 \
--epochs=3 \
--board_path="scratch"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000 \
--lr=0.01 \
--lam=0.0000001 \
--seed=2020 \
--dataset="ogbl-citation2" \
--recdim=128 \
--batch_size=1024 \
--epochs=2 \
--board_path="scratch"
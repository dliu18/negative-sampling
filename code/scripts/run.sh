# python main.py \
# --base_model="n2v" \
# --loss_func="sg_aug" \
# --test_set="test" \
# --n_negative=50 \
# --lr=0.01 \
# --lam=0.01 \
# --seed=2020 \
# --dataset="ogbl-ppa" \
# --recdim=128 \
# --batch_size=2 \
# --epochs=2

# ./scripts/hyperparameter-opt.sh Cora 4 128 2 hyperparam-fixed-split
# ./scripts/hyperparameter-opt.sh CiteSeer 4 128 2 hyperparam-fixed-split
# ./scripts/hyperparameter-opt.sh PubMed 4 128 32 hyperparam-fixed-split
# ./scripts/hyperparameter-opt.sh ogbl-collab 2 1024 128 hyperparam-fixed-split

# python main.py \
# --base_model="n2v" \
# --loss_func="sg" \
# --test_set="test" \
# --lr=0.1 \
# --seed=2020 \
# --dataset="ogbl-collab" \
# --recdim=128 \
# --batch_size=128 \
# --epochs=5 \
# --board_path="scratch"

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.1 \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=128 \
--epochs=50 \
--board_path="9-21-auc"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000 \
--lr=0.1 \
--lam=1 \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=1 \
--epochs=50 \
--board_path="9-21-auc"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.1 \
--lam=1 \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=1 \
--epochs=50 \
--board_path="9-21-auc"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.1 \
--lam=0.01 \
--seed=2020 \
--dataset="PubMed" \
--recdim=128 \
--batch_size=32 \
--epochs=50 \
--board_path="9-21-auc"

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.1 \
--n_negative=-1 \
--K=5 \
--alpha=0.75 \
--seed=2020 \
--dataset="ogbl-collab" \
--recdim=128 \
--batch_size=1024 \
--epochs=10 \
--board_path="9-21-auc"


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

# ./scripts/hyperparameter-opt.sh Cora 4 0.1 128 1 hyperparam
# ./scripts/hyperparameter-opt.sh CiteSeer 4 0.1 128 1 hyperparam
# ./scripts/hyperparameter-opt.sh PubMed 4 0.1 128 32 hyperparam
# ./scripts/hyperparameter-opt.sh ogbl-collab 3 0.1 1024 32 hyperparam

# python main.py \
# --base_model="n2v" \
# --loss_func="sg" \
# --test_set="test" \
# --lr=0.1 \
# --alpha=0.75 \
# --seed=2020 \
# --dataset="Cora" \
# --recdim=128 \
# --batch_size=128 \
# --epochs=1 \
# --board_path="scratch"

# python main.py \
# --base_model="n2v" \
# --loss_func="sg_aug" \
# --test_set="test" \
# --n_negative=10 \
# --lr=0.1 \
# --lam=0.1 \
# --seed=2020 \
# --dataset="Cora" \
# --recdim=128 \
# --batch_size=128 \
# --epochs=1 \
# --board_path="scratch"

# python main.py \
# --base_model="line" \
# --loss_func="sg" \
# --test_set="test" \
# --lr=0.1 \
# --seed=2020 \
# --dataset="Cora" \
# --recdim=128 \
# --batch_size=1 \
# --epochs=1 \
# --board_path="scratch"

# python main.py \
# --base_model="line" \
# --loss_func="sg_aug" \
# --test_set="test" \
# --n_negative=1000 \
# --lr=0.1 \
# --lam=1 \
# --seed=2020 \
# --dataset="Cora" \
# --recdim=128 \
# --batch_size=128 \
# --epochs=1 \
# --board_path="scratch"

# python main.py \
# --base_model="line" \
# --loss_func="sg_aug" \
# --test_set="test" \
# --n_negative=1000 \
# --lr=0.1 \
# --lam=0.001 \
# --seed=2020 \
# --dataset="ogbl-collab" \
# --recdim=128 \
# --batch_size=32 \
# --epochs=10 \
# --board_path="9-25-eval"

# python main.py \
# --base_model="line" \
# --loss_func="sg_aug" \
# --test_set="test" \
# --n_negative=1000 \
# --lr=0.1 \
# --lam=0.0001 \
# --seed=2020 \
# --dataset="ogbl-collab" \
# --recdim=128 \
# --batch_size=32 \
# --epochs=10 \
# --board_path="9-25-eval"

# python main.py \
# --base_model="line" \
# --loss_func="sg_aug" \
# --test_set="test" \
# --n_negative=1000 \
# --lr=0.01 \
# --lam=0.00001 \
# --seed=2020 \
# --dataset="ogbl-ppa" \
# --recdim=128 \
# --batch_size=256 \
# --epochs=3 \
# --board_path="9-25-eval"

# python main.py \
# --base_model="line" \
# --loss_func="sg_aug" \
# --test_set="test" \
# --n_negative=1000 \
# --lr=0.01 \
# --lam=0.000001 \
# --seed=2020 \
# --dataset="ogbl-ppa" \
# --recdim=128 \
# --batch_size=256 \
# --epochs=3 \
# --board_path="9-25-eval"

# python main.py \
# --base_model="line" \
# --loss_func="sg_aug" \
# --test_set="test" \
# --n_negative=1000000000 \
# --lr=0.1 \
# --lam=0.1 \
# --seed=2020 \
# --dataset="SBM-0.95-0.001" \
# --recdim=128 \
# --batch_size=128 \
# --epochs=2 \
# --board_path="scratch"

# python main.py \
# --base_model="line" \
# --loss_func="sg" \
# --test_set="test" \
# --lr=0.1 \
# --seed=2020 \
# --dataset="SBM-0.95-0.001" \
# --recdim=128 \
# --batch_size=128 \
# --epochs=2 \
# --board_path="scratch"

python main.py \
--base_model="line" \
--loss_func="sg" \
--test_set="test" \
--lr=0.001 \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=256 \
--epochs=15 \
--board_path="classifier"

# python main.py \
# --base_model="n2v" \
# --loss_func="sg" \
# --test_set="test" \
# --lr=0.01 \
# --seed=2020 \
# --dataset="ogbl-collab" \
# --recdim=128 \
# --batch_size=256 \
# --epochs=2 \
# --board_path="classifier" 
# --load=1 \
# --bypass_skipgram=True

# python main.py \
# --base_model="n2v" \
# --loss_func="sg" \
# --test_set="test" \
# --lr=0.01 \
# --n_negative=-1 \
# --K=5 \
# --alpha=0.75 \
# --seed=2020 \
# --dataset="ogbl-ppa" \
# --recdim=128 \
# --batch_size=256 \
# --epochs=5 \
# --board_path="classifier"

# python main.py \
# --base_model="n2v" \
# --loss_func="sg" \
# --test_set="test" \
# --lr=0.1 \
# --n_negative=-1 \
# --K=5 \
# --alpha=0.75 \
# --seed=2020 \
# --dataset="Cora" \
# --recdim=128 \
# --batch_size=128 \
# --epochs=20 \
# --board_path="classifier"
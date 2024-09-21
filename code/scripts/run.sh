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
# --batch_size=1024 \
# --epochs=2

./scripts/hyperparameter-opt.sh Cora 4 128 2 hyperparam-fixed-split
./scripts/hyperparameter-opt.sh CiteSeer 4 128 2 hyperparam-fixed-split
# ./scripts/hyperparameter-opt.sh PubMed 4 128 32 hyperparam-fixed-split
# ./scripts/hyperparameter-opt.sh ogbl-collab 2 1024 128 hyperparam-fixed-split
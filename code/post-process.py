import world
import utils
from model import SGModel
from world import cprint
import torch
import numpy as np
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

'''
Command line settings needed:
seed
latent_dim_rec
'''

datasets = ["ogbl-collab", "Cora"]
test_set = "test"

def get_n_negative(dataset_name, base_model, loss_func):
    n_negative_dict = {
        "Cora": {
            "line": 4, #FILL IN THIS VALUE
            "n2v": 2
        },
        "ogbl-collab": {
            "line": 2,
            "n2v": 2
        }
    }
    if loss_func == "sg":
        return 1
    elif loss_func == "no_neg":
        return 1000
    else 
        return n_negative_dict[dataset_name][base_model]

for dataset_name in datasets:
    dataset = None
    if dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = dataloader.SmallBenchmark(name=dataset_name, seed=world.seed)
    elif dataset_name in ["ogbl-ppa", "ogbl-collab", "ogbl-citation2"]:
        dataset = dataloader.OGBBenchmark(name=dataset_name, seed=world.seed)
    else:
        raise NotImplementedError(f"Haven't supported {dataset_name} yet!")

    sg_model = SGModel(world.config, dataset)
    sg_model = sg_model.to(world.device)

    for base_model in ["line", "n2v"]:
        for loss_func in ["sg", "sg_aug", "no_neg"]:
            n_negative = get_n_negative(dataset_name, base_model, loss_func)
            file = f"{base_model}-\
                    {loss_func if loss_func != "no_neg" else "no_neg"}-\
                    {n_negative}-\
                    {dataset_name}.pth.tar"
            weight_file = os.path.join(world.FILE_PATH,file)
            print(f"load and save to {weight_file}")
            try:
                sg_model.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
                world.cprint(f"loaded model weights from {weight_file}")
            except FileNotFoundError:
                print(f"{weight_file} not exists, start from beginning")

            _, all_mrr = Evaluator.test_mrr(sg_model, dataset, test_set)
            _, all_hits = Evaluator.test_hits(sg_model, dataset, test_set)
            print("\n")


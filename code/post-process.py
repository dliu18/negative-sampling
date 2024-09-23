import world
import utils
import dataloader
from evaluator import Evaluator
from model import SGModel
from world import cprint
import torch
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

'''
Command line settings needed:
seed
latent_dim_rec
'''

datasets = ["Cora", "CiteSeer", "PubMed", "ogbl-collab", "ogbl-ppa", "ogbl-citation2"]
metrics = ["Hits@K", "MRR"]
test_set = "test"

def get_loss_func_name(name):
    if name == "sg_neg":
        return "sg"
    elif name == "no_neg":
        return "sg_aug"
    return name
def get_n_negative(dataset_name, base_model, loss_func):
    n_negative_dict = {
        "Cora": {
            "line": 1000,
            "n2v": 10
        },
        "CiteSeer": {
            "line": 1000,
            "n2v": 10
        },
        "PubMed": {
            "line": 1000, 
            "n2v": 1000
        },
        "ogbl-collab": {
            "line": 1000,
            "n2v": 10
        },
        "ogbl-ppa": {
            "line": 1000,
            "n2v": 5
        },
        "ogbl-citation2": {
            "line": 1000,
            "n2v": 2
        }
    }
    if loss_func == "sg":
        return 10
    elif loss_func == "no_neg":
        return 1000000000
    else: 
        return n_negative_dict[dataset_name][base_model]

if __name__ == "__main__":
    fig, axs = plt.subplots(nrows = len(metrics), ncols = len(datasets), figsize=(7 * len(datasets), 7 * len(metrics)))

    for c_idx, dataset_name in enumerate(datasets):
        dataset = None
        if dataset_name in ["Cora", "CiteSeer", "PubMed"]:
            dataset = dataloader.SmallBenchmark(name=dataset_name, seed=world.seed)
        elif dataset_name in ["ogbl-ppa", "ogbl-collab", "ogbl-citation2"]:
            dataset = dataloader.OGBBenchmark(name=dataset_name, seed=world.seed)
        else:
            raise NotImplementedError(f"Haven't supported {dataset_name} yet!")

        degrees = dataset.degrees
        print(degrees.shape)
        print(dataset.n_users)
        sg_model = SGModel(world.config, dataset)
        sg_model = sg_model.to(world.device)

        for base_model in ["n2v"]:
            for loss_func in ["sg", "sg_neg", "sg_aug", "no_neg"]:
                n_negative = get_n_negative(dataset_name, base_model, loss_func)
                loss_func_name = get_loss_func_name(loss_func)
                file = f"{base_model}-{loss_func_name}-{n_negative}-{dataset_name}.pth.tar"
                weight_file = join(world.FILE_PATH,file)
                print(f"load and save to {weight_file}")
                try:
                    sg_model.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
                    world.cprint(f"loaded model weights from {weight_file}")
                except FileNotFoundError:
                    print(f"{weight_file} not exists, start from beginning")

                eval_edges = dataset.get_eval_data(test_set)
                _, all_mrr = Evaluator.test_mrr(sg_model, dataset, test_set)
                _, all_hits = Evaluator.test_hits(sg_model, dataset, test_set)
                
                eval_degrees = degrees[eval_edges[0]]
                percentiles = torch.quantile(eval_degrees, q = torch.arange(0, 1.02, 0.02).to(world.device))
                x, y_hits, y_mrr = [], [], []
                for idx in range(1, len(percentiles)):
                    x.append(percentiles[idx].cpu())
                    low = percentiles[idx - 1]
                    high = percentiles[idx]
                    mask = ((eval_degrees >= low) & (eval_degrees < high)).reshape(1, -1)
                    y_hits.append(all_hits[mask].mean().cpu())
                    y_mrr.append(all_mrr[mask].mean().cpu())
                axs[0][c_idx].plot(x, y_hits, linewidth=2, label = f"{base_model} {loss_func}")
                axs[1][c_idx].plot(x, y_mrr, linewidth=2, label = f"{base_model} {loss_func}")

    for r in range(len(axs)):
        for c in range(len(axs[r])):
            axs[r][c].grid()
            axs[r][c].legend()
            axs[r][c].set_xlabel("Degree")
            axs[r][c].set_ylabel(metrics[r])
            axs[r][c].set_title(datasets[c])
            axs[r][c].set_xscale("log")
    plt.savefig("../figs/metrics_by_degree.pdf", bbox_inches="tight")
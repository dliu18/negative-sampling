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

datasets = ["Cora", "CiteSeer", "PubMed", "ogbl-collab", "ogbl-ppa", "ogbl-citation2"]
# datasets = ["ogbl-collab", "ogbl-ppa", "ogbl-citation2"]

metrics = ["Hits@K", "MRR"]
test_set = "test"
base_model = "line"
K = 50

def color(name):
    colors = {
        "sg": "#377eb8", 
        "sg_weighted": "#e41a1c",
        "sg_aug": "#984ea3",
        "sg_aug_weighted": "#ff7f00",
        "no_neg": "#4daf4a",
    }
    return colors[name]

def display_name(name):
    names = {
        "n2v": "node2vec",
        "line": "LINE",
        "sg": "I",
        "sg_weighted": "I (\u03B1 = 0.75)",
        "sg_aug": "II",
        "sg_aug_weighted": "II (\u03B1 = 0.75)",
        "no_neg": " No Negative"
    }
    return names[name]

def get_loss_func_name(name):
    if name == "sg_weighted":
        return "sg"
    elif name == "sg_aug_weighted":
        return "sg_aug"
    elif name == "no_neg":
        return "sg_aug"
    return name
def get_n_negative(dataset_name, base_model, loss_func):
    n_negative_dict = {
        "Cora": {
            "line": 100,
            "n2v": 10
        },
        "CiteSeer": {
            "line": 100,
            "n2v": 100
        },
        "PubMed": {
            "line": 100, 
            "n2v": 100
        },
        "ogbl-collab": {
            "line": 10,
            "n2v": 10
        },
        "ogbl-ppa": {
            "line": 10,
            "n2v": 10
        },
        "ogbl-citation2": {
            "line": 10,
            "n2v": 10
        }
    }
    if loss_func == "sg":
        return 10
    elif loss_func == "sg_weighted" or loss_func == "sg_aug_weighted":
        return -1
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
        cluster_coefs = torch.Tensor(dataset.get_clustering_coefs()).to(world.device)
        sg_model = SGModel(world.config, dataset)
        sg_model = sg_model.to(world.device)

        # for loss_func in ["sg", "sg_weighted", "sg_aug", "sg_aug_weighted", "no_neg"]:
        for loss_func in ["sg", "sg_aug", "no_neg"]:
            print(f"{base_model} {loss_func}")
            n_negative = get_n_negative(dataset_name, base_model, loss_func)
            loss_func_name = get_loss_func_name(loss_func)
            file = f"{base_model}-{loss_func_name}-{n_negative}-{dataset_name}.pth.tar"
            weight_file = join(world.FILE_PATH, file)
            print(f"load and save to {weight_file}")
            try:
                sg_model.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
                world.cprint(f"loaded model weights from {weight_file}")
            except FileNotFoundError:
                print(f"{weight_file} not exists, start from beginning")
                continue
            except:
                continue
            eval_edges = dataset.get_eval_data(test_set)
            _, all_mrr = Evaluator.test_mrr(sg_model, dataset, test_set, use_classifier=True)
            _, all_hits = Evaluator.test_hits(sg_model, K, dataset, test_set, use_classifier=True)
            
            # features = degrees[eval_edges[0]]
            features = cluster_coefs[eval_edges[0]]
            percentiles = torch.quantile(features, q = torch.arange(0, 1.02, 0.02).to(world.device))
            x, y_hits, y_mrr = [], [], []
            for idx in range(1, len(percentiles)):
                x.append(percentiles[idx].cpu())
                low = percentiles[idx - 1]
                high = percentiles[idx]
                mask = ((features >= low) & (features <= high)).reshape(1, -1)
                y_hits.append(all_hits[mask].mean().cpu())
                y_mrr.append(all_mrr[mask].mean().cpu())
            axs[0][c_idx].plot(x, y_hits, 
                linewidth=2, 
                label = f"{display_name(base_model)} {display_name(loss_func)}",
                color = color(loss_func))
            axs[1][c_idx].plot(x, y_mrr, 
                linewidth=2, 
                label = f"{display_name(base_model)} {display_name(loss_func)}",
                color = color(loss_func))

    for r in range(len(axs)):
        for c in range(len(axs[r])):
            axs[r][c].grid()
            axs[r][c].legend()
            axs[r][c].set_xlabel("Clustering Coefficient")
            axs[r][c].set_ylabel(metrics[r])
            axs[r][c].set_title(datasets[c])
            # axs[r][c].set_xscale("log")
    plt.savefig(f"../figs/post-rebuttal/metrics_by_clustering_coef_{base_model}.pdf", bbox_inches="tight")
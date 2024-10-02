import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dataloader import SmallBenchmark, OGBBenchmark

plt.rcParams.update({
    'font.size': 14,        # Default text font size
    'axes.titlesize': 18,   # Title font size
    'axes.labelsize': 16,   # X and Y label font size
    'xtick.labelsize': 14,  # X-axis tick label font size
    'ytick.labelsize': 14,  # Y-axis tick label font size
    'legend.fontsize': 14   # Legend font size
})

if __name__ == "__main__":
	df = pd.read_csv("../outputs/9-25-eval-aggregated.csv")
	df["AUC-ROC"] = df["metrics/test/AUC_ROC"]
	df["MRR"] = df["metrics/test/MRR"]
	df["Hits@k"] = df["metrics/test/Hits_50"]

	df = df[df["Model"] == "line"]

	cols = ["Graph", "Loss Function", "n_negative", "AUC-ROC", "MRR", "Hits@k"]

	df = df[cols].set_index(["Graph", "Loss Function", "n_negative"])


	fig, axs = plt.subplots(ncols = 3, figsize = (18, 5), sharex = True, sharey = True)

	for name in ["Cora", "CiteSeer", "PubMed"]:
		dataset = SmallBenchmark(name, seed = 2020)
		avg_clustering = np.mean(dataset.get_clustering_coefs())

		orig_metrics = df.loc[name, "sg", 10]
		no_neg_metrics = df.loc[name, "sg_aug", 1000000000]

		auc_gain = (no_neg_metrics["AUC-ROC"] - orig_metrics["AUC-ROC"]) 
		if not np.isnan(auc_gain):
			axs[0].scatter(avg_clustering, auc_gain, label = name, s = 150)

		mrr_gain = (no_neg_metrics["MRR"] - orig_metrics["MRR"]) 
		if not np.isnan(mrr_gain):
			axs[1].scatter(avg_clustering, mrr_gain, label = name, s = 150)

		hits_gain = (no_neg_metrics["Hits@k"] - orig_metrics["Hits@k"])
		if not np.isnan(hits_gain):
			axs[2].scatter(avg_clustering, hits_gain, label = name, s = 150)


	for name in ["ogbl-collab", "ogbl-ppa", "ogbl-citation2"]:
		dataset = OGBBenchmark(name, seed = 2020)
		avg_clustering = np.mean(dataset.get_clustering_coefs())

		orig_metrics = df.loc[name, "sg", 10]
		no_neg_metrics = df.loc[name, "sg_aug", 1000000000]

		auc_gain = (no_neg_metrics["AUC-ROC"] - orig_metrics["AUC-ROC"])
		if not np.isnan(auc_gain): 
			axs[0].scatter(avg_clustering, auc_gain, label = name, s = 150)

		mrr_gain = (no_neg_metrics["MRR"] - orig_metrics["MRR"]) 
		if not np.isnan(mrr_gain):
			axs[1].scatter(avg_clustering, mrr_gain, label = name, s = 150)

		hits_gain = (no_neg_metrics["Hits@k"] - orig_metrics["Hits@k"])
		if not np.isnan(hits_gain):
			axs[2].scatter(avg_clustering, hits_gain, label = name, s = 150)

	metric_names = ["AUC-ROC", "MRR", "Hits@50"]
	for idx, metric_name in enumerate(metric_names):
		axs[idx].set_xlabel("Average clustering coefficient")
		axs[idx].set_ylabel(f"{metric_name} " + r"$\Delta$")
		axs[idx].set_title(metric_name)
		axs[idx].grid()

	handles, labels = axs[0].get_legend_handles_labels()
	fig.legend(handles, labels, bbox_to_anchor = (0.5, 1.15), loc='upper center', ncols = len(labels))
	fig.subplots_adjust(top=0.97)

	fig.savefig("../figs/agg_performance_by_clustering_coef.pdf", bbox_inches="tight")
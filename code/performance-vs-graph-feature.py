import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dataloader import SmallBenchmark, OGBBenchmark

import fig_config

SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 26
plt.rc('font', size=SMALL_SIZE, family = "Nimbus Roman")          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



if __name__ == "__main__":
	df = pd.read_csv("../outputs/kdd25/summary-kdd-25-2.csv")
	df["AUC-ROC"] = df["metrics/test/AUC_ROC"]
	df["MRR"] = df["metrics/test/MRR"]
	df["Hits@100"] = df["metrics/test/Hits_100"]

	df = df[df["Model"] == "line"]

	cols = ["Graph", "Loss Function", "n_negative", "AUC-ROC", "MRR", "Hits@100"]

	df = df[cols].set_index(["Graph", "Loss Function", "n_negative"])


	fig, axs = plt.subplots(ncols = 3, figsize = (18, 5), sharex = True, sharey = True)

	for name in ["Cora", "CiteSeer", "PubMed"]:
		dataset = SmallBenchmark(name, seed = 2020, test_set="test", test_set_frac=0.2)
		avg_clustering = np.mean(dataset.get_clustering_coefs())

		try:
			orig_metrics = df.loc[name, "sg", 10]
			no_neg_metrics = df.loc[name, "sg_aug", 1000000000]
		except:
			continue

		auc_gain = (no_neg_metrics["AUC-ROC"] - orig_metrics["AUC-ROC"]) 
		if not np.isnan(auc_gain):
			axs[0].scatter(avg_clustering, auc_gain, label = name, s = 150)

		mrr_gain = (no_neg_metrics["MRR"] - orig_metrics["MRR"]) 
		if not np.isnan(mrr_gain):
			axs[1].scatter(avg_clustering, mrr_gain, label = name, s = 150)

		hits_gain = (no_neg_metrics["Hits@100"] - orig_metrics["Hits@100"])
		if not np.isnan(hits_gain):
			axs[2].scatter(avg_clustering, hits_gain, label = name, s = 150)


	for name in ["ogbl-collab", "ogbl-ppa", "ogbl-citation2", "ogbl-vessel"]:
		dataset = OGBBenchmark(name, seed = 2020, test_set="test")
		avg_clustering = np.mean(dataset.get_clustering_coefs())

		try:
			orig_metrics = df.loc[name, "sg", 10]
			no_neg_metrics = df.loc[name, "sg_aug", 1000000000]
		except:
			continue

		auc_gain = (no_neg_metrics["AUC-ROC"] - orig_metrics["AUC-ROC"])
		if not np.isnan(auc_gain): 
			axs[0].scatter(avg_clustering, auc_gain, label = name, s = 150)

		mrr_gain = (no_neg_metrics["MRR"] - orig_metrics["MRR"]) 
		if not np.isnan(mrr_gain):
			axs[1].scatter(avg_clustering, mrr_gain, label = name, s = 150)

		hits_gain = (no_neg_metrics["Hits@100"] - orig_metrics["Hits@100"])
		if not np.isnan(hits_gain):
			axs[2].scatter(avg_clustering, hits_gain, label = name, s = 150)

	metric_names = ["AUC-ROC", "MRR", "Hits@100"]
	for idx, metric_name in enumerate(metric_names):
		axs[idx].set_xlabel("Average clustering coefficient")
		axs[idx].set_ylabel(f"{metric_name} " + r"$\Delta$")
		axs[idx].set_title(metric_name)
		axs[idx].grid()

	handles, labels = axs[0].get_legend_handles_labels()
	fig.legend(handles, labels, bbox_to_anchor = (0.5, 1.2), loc='upper center', ncols = len(labels))
	fig.subplots_adjust(top=0.97)

	fig.savefig("../figs/kdd25/agg_performance_by_clustering_coef.pdf", bbox_inches="tight")
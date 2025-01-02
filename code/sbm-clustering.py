import pandas as pd
import matplotlib.pyplot as plt
from dataloader import SmallBenchmark
import numpy as np 

plt.rcParams.update({
    'font.size': 14,        # Default text font size
    'axes.titlesize': 16,   # Title font size
    'axes.labelsize': 14,   # X and Y label font size
    'xtick.labelsize': 14,  # X-axis tick label font size
    'ytick.labelsize': 14,  # Y-axis tick label font size
    'legend.fontsize': 12   # Legend font size
})

key = "metrics/test/AUC_ROC"
if __name__ == "__main__":
	fig, ax = plt.subplots()

	# df = pd.read_csv("../../outputs/summary-sbm-final.csv")
	df = pd.read_csv("../outputs/post-rebuttal/summary-sbm-line-extended-mlp.csv")

	df = df[df['Model'] == 'line']
	df["p"] = [float(row.split("-")[1]) for row in df["Graph"]]
	df["q"] = [float(row.split("-")[2]) for row in df["Graph"]]
	df["p/q"] = df["p"] / df["q"]

	df = df.sort_values("p/q")
	df = df[["Loss Function", "n_negative", "p", "q", key, "p/q"]]
	print(df)
	# clustering_coefs = []
	# for i in range(len(df)):
	# 	p = df.iloc[i]["p"]
	# 	q = df.iloc[i]["q"]
	# 	dataset = SmallBenchmark(name = f"SBM-{p}-{q}", seed = 2020)
	# 	clustering_coefs.append(np.mean(dataset.get_clustering_coefs()))
	# df["clustering"] = clustering_coefs

	mask = (df["Loss Function"] == "sg") & (df["n_negative"] >= 0)
	x = df[mask]["p/q"]
	y = df[mask][key]
	ax.plot(x, y, label = "LINE I", color = "#377eb8", linewidth = 2)

	# mask = (df["Loss Function"] == "sg") & (df["n_negative"] == -1)
	# x = df[mask]["q/p"]
	# y = df[mask][key]
	# ax.plot(x, y, label = "LINE I (\u03B1 = 0.75)", color = "#e41a1c")

	mask = (df["Loss Function"] == "sg_aug") & (df["n_negative"] <= 1000)
	x = df[mask]["p/q"]
	y = df[mask][key]
	ax.plot(x, y, label = "LINE II", color = "#984ea3", linewidth = 2)

	# mask = (df["Loss Function"] == "sg_aug") & (df["n_negative"] == -1)
	# x = df[mask]["q/p"]
	# y = df[mask][key]
	# ax.plot(x, y, label = "LINE II (\u03B1 = 0.75)", color = "#ff7f00")

	mask = (df["Loss Function"] == "sg_aug") & (df["n_negative"] == 1000000000)
	x = df[mask]["p/q"]
	y = df[mask][key]
	ax.plot(x, y, label = "LINE II" + r"$^0$", color = "#4daf4a", linewidth = 2)

	ax.legend()
	ax.grid()
	ax.set_xlabel("Within-block / Between-block edge probability")
	ax.set_ylabel("AUC-ROC")
	ax.set_xscale("log")
	fig.savefig("../figs/post-rebuttal/sbm_series-line.pdf", bbox_inches = "tight")	
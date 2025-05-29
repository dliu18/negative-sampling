import world
import pandas as pd
import matplotlib.pyplot as plt
from dataloader import SmallBenchmark
import numpy as np 

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE, family = "Nimbus Roman")          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


key = "metrics/test/AUC_ROC"
alg = world.config["base_model"]
num_trials = 8
assert alg in ["n2v", "line"]
alg_display_name = "node2vec" if alg=="n2v" else "LINE"

if __name__ == "__main__":
	fig, ax = plt.subplots()

	df = pd.read_csv("../outputs/kdd25/summary-sbm-iclr-half.csv")
	df = df[df['Model'] == alg]
	df["p"] = [float(row.split("-")[1]) for row in df["Graph"]]
	df["q"] = [float(row.split("-")[2]) for row in df["Graph"]]
	df["p/q"] = df["p"] / df["q"]

	df = df.sort_values("p/q")
	df = df[["Loss Function", "n_negative", "p", "q", key, "p/q"]]
	df = df[df[key].notna()]
	
	# dfs_for_each_trial = []
	# for trial_idx in range(1, num_trials + 1):
	# 	df = pd.read_csv(f"../outputs/kdd25/sbm-trials/summary-sbm-iclr-half-{trial_idx}.csv")

	# 	df = df[df['Model'] == alg]
	# 	df["p"] = [float(row.split("-")[1]) for row in df["Graph"]]
	# 	df["q"] = [float(row.split("-")[2]) for row in df["Graph"]]
	# 	df["p/q"] = df["p"] / df["q"]

	# 	df = df.sort_values("p/q")
	# 	df = df[["Loss Function", "n_negative", "p", "q", key, "p/q"]]
	# 	df = df[df[key].notna()]
	# 	dfs_for_each_trial.append(df)

	# df = pd.concat(dfs_for_each_trial)
	# agg_df = df.groupby(["p/q", "Loss Function", "n_negative"]).agg(
	# 	p=("p", "first"),
	# 	q=("q", "first"),
	#     mean_value=(key, "mean"),
	#     min_value=(key, "min"),
	#     max_value=(key, "max"),
	#     count=(key, "size")
	# ).reset_index()
	# df = agg_df


	# to plot the aggregated results, change the key to "mean_value"
	mask = (df["Loss Function"] == "sg") & (df["n_negative"] >= 0)
	x = df[mask]["p/q"]
	y = df[mask][key]
	ax.plot(x, y, label = f"{alg_display_name} I", color = "#377eb8", linewidth = 2)

	mask = (df["Loss Function"] == "sg_aug") & (df["n_negative"] == 10)
	x = df[mask]["p/q"]
	y = df[mask][key]
	ax.plot(x, y, label = f"{alg_display_name} II", color = "#984ea3", linewidth = 2)

	mask = (df["Loss Function"] == "sg_aug") & (df["n_negative"] == 1000000000)
	x = df[mask]["p/q"]
	y = df[mask][key]
	ax.plot(x, y, label = f"{alg_display_name} II" + r"$^0$", color = "#4daf4a", linewidth = 2)

	ax.legend()
	ax.grid()
	ax.set_xlabel("Pr(within-block edge) / Pr(between-block edge)")
	# ax.set_xlabel("Within-block edge probability")
	ax.set_ylabel("AUC-ROC")
	ax.set_xscale("log")
	fig.savefig(f"../figs/kdd25/sbm-{alg}.pdf", bbox_inches = "tight")	
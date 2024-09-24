import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
	fig, ax = plt.subplots()

	df = pd.read_csv("../../outputs/summary-sbm.csv")
	df["p"] = [float(row.split("-")[1]) for row in df["Graph"]]
	df["q"] = [float(row.split("-")[2]) for row in df["Graph"]]
	df["q/p"] = df["q"] / df["p"]

	mask = (df["Loss Function"] == "sg") & (df["n_negative"] == -1)
	x = df[mask]["q/p"]
	y = df[mask]["metrics/AUC_ROC"]
	ax.plot(x, y, label = "node2vec (\u03B1 = 0.75)", color = "#e41a1c")

	mask = (df["Loss Function"] == "sg") & (df["n_negative"] >= 0)
	x = df[mask]["q/p"]
	y = df[mask]["metrics/AUC_ROC"]
	ax.plot(x, y, label = "node2vec I", color = "#377eb8")

	mask = (df["Loss Function"] == "sg_aug") & (df["n_negative"] == 1000000)
	x = df[mask]["q/p"]
	y = df[mask]["metrics/AUC_ROC"]
	ax.plot(x, y, label = "node2vec No Negative", color = "#4daf4a")

	mask = (df["Loss Function"] == "sg_aug") & (df["n_negative"] <= 1000)
	x = df[mask]["q/p"]
	y = df[mask]["metrics/AUC_ROC"]
	ax.plot(x, y, label = "node2vec II", color = "#984ea3")

	ax.legend()
	ax.grid()
	ax.set_xlabel("q / p")
	ax.set_ylabel("AUC-ROC")
	fig.savefig("../../figs/sbm_series.pdf", bbox_inches = "tight")	
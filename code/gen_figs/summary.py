import pandas as pd

df = pd.read_csv("../../outputs/9-25-eval-aggregated.csv")

mask = (df["n_negative"] >= 0) & (df["n_negative"] < 1000000000)
df = df[mask]

df["AUC"] = df["metrics/test/AUC_ROC"]
df["Memory"] = df["GPU_usage/memory__bytes_"]

cols = ["Graph", "Model", "Loss Function", "Duration", "Memory", "AUC"]

df = df[cols]

diffs = []

graphs = ["CiteSeer", "Cora", "PubMed", "ogbl-collab", "ogbl-ppa", "ogbl-citation2"]
models = ["n2v", "line"]

for model in models:
	for graph in graphs:
		mask = (df["Graph"] == graph) & (df["Model"] == model)
		df_subset = df[mask].set_index("Loss Function")

		assert len(df_subset) == 2
		diffs.append({
			"Graph": graph,
			"Model": model,
			"Duration": 100 * (df_subset.loc["sg_aug"]["Duration"] - df_subset.loc["sg"]["Duration"]) / df_subset.loc["sg"]["Duration"],
			"Memory": 100 * (df_subset.loc["sg_aug"]["Memory"] - df_subset.loc["sg"]["Memory"]) / df_subset.loc["sg"]["Memory"],
			"AUC": 100 * (df_subset.loc["sg_aug"]["AUC"] - df_subset.loc["sg"]["AUC"]) / df_subset.loc["sg"]["AUC"],
			})
diffs_df = pd.DataFrame(diffs)
diffs_df.to_csv("../../outputs/diff.csv")

print(diffs_df)

print("Memory: ", diffs_df["Memory"].min())
print("Duration: ", diffs_df["Duration"].min())
print("AUC: ", diffs_df["AUC"].mean())
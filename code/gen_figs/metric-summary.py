import pandas as pd

def extract_metric(df_subset, model_symbol, metric_name):
	if model_symbol == "I":
		return df_subset.loc["sg"].iloc[0][metric_name]
	elif model_symbol == "II":
		row = df_subset.loc["sg_aug"].iloc[0]
		assert row.name < 1000000000
		return row[metric_name]
	elif model_symbol == "II'":
		return df_subset.loc["sg_aug"].loc[1000000000][metric_name]

df = pd.read_csv("../../outputs/9-25-eval-aggregated.csv")

mask = (df["n_negative"] >= 0) 
df = df[mask]

df["Time"] = df["Duration"] / 60
df["AUC-ROC"] = df["metrics/test/AUC_ROC"]
df["Memory"] = df["GPU_usage/memory__bytes_"] / 1e9

cols = ["Graph", "Model", "Loss Function", "n_negative", "Time", "Memory", "AUC-ROC"]

df = df[cols]

summary = []

graphs = ["CiteSeer", "Cora", "PubMed", "ogbl-collab", "ogbl-ppa", "ogbl-citation2"]
models = ["n2v", "line"]

for model in models:
	for graph in graphs:
		mask = (df["Graph"] == graph) & (df["Model"] == model)
		df_subset = df[mask].set_index(["Loss Function", "n_negative"])
		# print(df_subset)

		assert len(df_subset) == 3
		row = {"Graph": graph, "Model": model}
		for metric in ["Time", "Memory", "AUC-ROC"]:
			for model_variant in ["I", "II'", "II"]:
				row[f"{metric} {model_variant}"] = f"{extract_metric(df_subset, model_variant, metric):.2f} "
			delta = 100 * (float(row[f"{metric} II"]) - float(row[f"{metric} I"])) / float(row[f"{metric} I"])
			# row[f"{metric} Delta"] = f"({delta:.1f}\%) "
			row[f"{metric} Delta"] = delta
		# print(row)
		summary.append(row)

summary_df = pd.DataFrame(summary)
summary_df.to_csv("../../outputs/metric_summary.csv", index = False, sep = ",")
# diffs_df = pd.DataFrame(diffs)
# diffs_df.to_csv("../../outputs/diff.csv")

# print(diffs_df)

# print("Memory: ", diffs_df["Memory"].min())
# print("Duration: ", diffs_df["Duration"].min())
# print("AUC: ", diffs_df["AUC"].mean())
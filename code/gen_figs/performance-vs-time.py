import pandas as pd 
import matplotlib.pyplot as plt

current_alg = "n2v"

graph_to_idx = {
	"Cora": 0,
	"CiteSeer": 1,
	"PubMed": 2,
	"ogbl-collab": 3,
	"ogbl-ppa": 4,
	"ogbl-citation2": 5
}

def get_display_name(alg_name, loss_func, n_negative):
	model_name = ""
	if alg_name == "n2v":
		model_name = "node2vec"
	elif alg_name == "line":
		model_name = "LINE"

	if loss_func == "sg":
		if n_negative == -1:
			return model_name + " (\u03B1 = 0.75)"
		else: 
			return model_name + " I"
	elif loss_func == "sg_aug":
		if n_negative == 1000000000:
			return model_name + " No Negative"
		else:
			return model_name + " II"

def get_marker(loss_func, n_negative):
	return "o"

	# if loss_func == "sg":
	# 	if n_negative == -1:
	# 		return "o"
	# 	else: 
	# 		return "D"
	# elif loss_func == "sg_aug":
	# 	if n_negative == 1000000000:
	# 		return "X"
	# 	else:
	# 		return "*"

def get_color(alg_name):
	if loss_func == "sg":
		if n_negative == -1:
			return "#e41a1c"
		else: 
			return "#377eb8"
	elif loss_func == "sg_aug":
		if n_negative == 1000000000:
			return "#4daf4a"
		else:
			return "#984ea3"

results_file = "../../outputs/summary-9-21-auc.csv"

if __name__ == "__main__":
	df = pd.read_csv(results_file)
	fig_auc, axs_auc = plt.subplots(
		ncols = len(graph_to_idx.keys()),
		figsize = (7 * len(graph_to_idx.keys()), 7),
		sharey = True)
	fig_mrr, axs_mrr = plt.subplots(
		ncols = len(graph_to_idx.keys()),
		figsize = (7 * len(graph_to_idx.keys()), 7),
		sharey = True)
	fig_hits, axs_hits = plt.subplots(
		ncols = len(graph_to_idx.keys()),
		figsize = (7 * len(graph_to_idx.keys()), 7),
		sharey = True)

	for idx in range(len(df)):
		graph_name = df.iloc[idx]["Graph"]
		alg_name = df.iloc[idx]["Model"]
		loss_func = df.iloc[idx]["Loss Function"]
		n_negative = df.iloc[idx]["n_negative"]
		auc = df.iloc[idx]["metrics/AUC_ROC"]
		mrr = df.iloc[idx]["metrics/MRR"]
		hits = df.iloc[idx]["metrics/Hits_50"]

		if df.iloc[idx]["Steps"] <= 1:
			continue

		if alg_name != current_alg:
			continue

		col_idx = graph_to_idx[graph_name]
		duration = float(df.iloc[idx]["Duration"].replace(",", ""))
		axs_auc[col_idx].scatter(x = [duration],
			y = [auc],
			marker = get_marker(loss_func, n_negative),
			label = get_display_name(alg_name, loss_func, n_negative),
			color = get_color(alg_name),
			s = 100)
		axs_mrr[col_idx].scatter(x = [duration],
			y = [mrr],
			marker = get_marker(loss_func, n_negative),
			label = get_display_name(alg_name, loss_func, n_negative),
			color = get_color(alg_name),
			s = 100)
		axs_hits[col_idx].scatter(x = [duration],
			y = [hits],
			marker = get_marker(loss_func, n_negative),
			label = get_display_name(alg_name, loss_func, n_negative),
			color = get_color(alg_name),
			s = 100)

for idx, ax in enumerate(axs_auc):
	ax.set_xlabel("Training Time (s)")
	ax.set_ylabel("AUC-ROC")
	ax.grid()
	ax.legend()
	ax.set_title(list(graph_to_idx.keys())[idx] + " AUC-ROC")
fig_auc.savefig(f"../../figs/auc_vs_time_{current_alg}.pdf", bbox_inches = "tight")

for idx, ax in enumerate(axs_mrr):
	ax.set_xlabel("Training Time (s)")
	ax.set_ylabel("MRR")
	ax.grid()
	ax.legend()
	ax.set_title(list(graph_to_idx.keys())[idx] + " MRR")
fig_auc.savefig(f"../../figs/mrr_vs_time_{current_alg}.pdf", bbox_inches = "tight")

for idx, ax in enumerate(axs_hits):
	ax.set_xlabel("Training Time (s)")
	ax.set_ylabel("Hits@50")
	ax.legend()
	ax.grid()
	ax.set_title(list(graph_to_idx.keys())[idx] + " Hits@50")
fig_auc.savefig(f"../../figs/hits_vs_time_{current_alg}.pdf", bbox_inches = "tight")
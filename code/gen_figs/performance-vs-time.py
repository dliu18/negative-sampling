import pandas as pd 
import matplotlib.pyplot as plt
from sys import argv
import fig_config

current_alg = argv[1]
xaxis = argv[2]

xlabel = ""
if xaxis == "time":
	xlabel = "Relative Training Time"
elif xaxis == "memory":
	xlabel = "Relative Max GPU Memory (GB)"

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
			return model_name + " I (\u03B1 = 0.75)"
		else: 
			return model_name + " I"
	elif loss_func == "sg_aug":
		if n_negative == 1000000000:
			return model_name + r" II$^0$"
		elif n_negative == -1:
			return model_name + " II (\u03B1 = 0.75)"
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
		elif n_negative == -1:
			return "#ff7f00"
		else:
			return "#984ea3"

results_file = "../../outputs/9-25-eval-aggregated.csv"

if __name__ == "__main__":
	df = pd.read_csv(results_file)
	fig_auc, axs_auc = plt.subplots(
		ncols = len(graph_to_idx.keys()),
		figsize = (7 * len(graph_to_idx.keys()), 5.5),
		sharey = True,
		sharex = True)
	fig_mrr, axs_mrr = plt.subplots(
		ncols = len(graph_to_idx.keys()),
		figsize = (7 * len(graph_to_idx.keys()), 5.5),
		sharey = True,
		sharex = True)
	fig_hits, axs_hits = plt.subplots(
		ncols = len(graph_to_idx.keys()),
		figsize = (7 * len(graph_to_idx.keys()), 5.5),
		sharey = True,
		sharex = True)

	for idx in range(len(df)):
		graph_name = df.iloc[idx]["Graph"]
		alg_name = df.iloc[idx]["Model"]
		loss_func = df.iloc[idx]["Loss Function"]
		n_negative = df.iloc[idx]["n_negative"]
		auc = df.iloc[idx]["metrics/test/AUC_ROC"]
		mrr = df.iloc[idx]["metrics/test/MRR"]
		hits = df.iloc[idx]["metrics/test/Hits_50"]

		if df.iloc[idx]["Steps"] <= 1:
			continue

		if alg_name != current_alg:
			continue

		col_idx = graph_to_idx[graph_name]

		max_duration = df[(df["Graph"] == graph_name) & (df["Model"] == alg_name)]["Duration"].max()
		max_mem = df[(df["Graph"] == graph_name) & (df["Model"] == alg_name)]["GPU_usage/memory__bytes_"].max()
		duration = float(df.iloc[idx]["Duration"]) / max_duration
		memory = float(df.iloc[idx]["GPU_usage/memory__bytes_"] / max_mem)
		x = duration if xaxis == "time" else memory

		axs_auc[col_idx].scatter(x = [x],
			y = [auc],
			marker = get_marker(loss_func, n_negative),
			label = get_display_name(alg_name, loss_func, n_negative),
			color = get_color(alg_name),
			s = 100)
		axs_mrr[col_idx].scatter(x = [x],
			y = [mrr],
			marker = get_marker(loss_func, n_negative),
			label = get_display_name(alg_name, loss_func, n_negative),
			color = get_color(alg_name),
			s = 100)
		axs_hits[col_idx].scatter(x = [x],
			y = [hits],
			marker = get_marker(loss_func, n_negative),
			label = get_display_name(alg_name, loss_func, n_negative),
			color = get_color(alg_name),
			s = 100)

for idx, ax in enumerate(axs_auc):
	ax.set_xlabel(xlabel)
	ax.set_ylabel("AUC-ROC")
	ax.grid()
	ax.set_title(list(graph_to_idx.keys())[idx] + " AUC-ROC")
handles, labels = axs_auc[0].get_legend_handles_labels()
fig_auc.legend(handles, labels, bbox_to_anchor = (0.5, 1.15), loc='upper center', ncols = len(labels))
fig_auc.subplots_adjust(top=0.95)
fig_auc.savefig(f"../../figs/auc_vs_{xaxis}_{current_alg}.pdf", bbox_inches = "tight")

for idx, ax in enumerate(axs_mrr):
	ax.set_xlabel(xlabel)
	ax.set_ylabel("MRR")
	ax.grid()
	ax.set_title(list(graph_to_idx.keys())[idx] + " MRR")
handles, labels = axs_mrr[0].get_legend_handles_labels()
fig_mrr.legend(handles, labels, bbox_to_anchor = (0.5, 1.15), loc='upper center', ncols = len(labels))
fig_mrr.subplots_adjust(top=0.95)
fig_mrr.savefig(f"../../figs/mrr_vs_{xaxis}_{current_alg}.pdf", bbox_inches = "tight")

for idx, ax in enumerate(axs_hits):
	ax.set_xlabel(xlabel)
	ax.set_ylabel("Hits@50")
	ax.grid()
	ax.set_title(list(graph_to_idx.keys())[idx] + " Hits@50")
handles, labels = axs_hits[0].get_legend_handles_labels()
fig_hits.legend(handles, labels, bbox_to_anchor = (0.5, 1.15), loc='upper center', ncols = len(labels))
fig_hits.subplots_adjust(top=0.95)
fig_hits.savefig(f"../../figs/hits_vs_{xaxis}_{current_alg}.pdf", bbox_inches = "tight")
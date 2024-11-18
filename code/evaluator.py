import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import BasicDataset
from model import BasicModel
from sklearn.metrics import roc_auc_score 

EVAL_BATCH_SIZE = 128

class Evaluator:
	@staticmethod
	@torch.no_grad()
	def test_auc(sg_model, dataset, test_set):
		sg_model.eval()
		eval_edges_pos = dataset.get_eval_data(test_set)
		eval_edges_neg = dataset.get_roc_negatives(test_set)
		assert abs(eval_edges_pos.size(1) - eval_edges_neg.size(1)) / eval_edges_pos.size(1) < 0.1

		eval_edges = torch.cat([eval_edges_pos, eval_edges_neg], dim = 1)
		
		num_pos = eval_edges_pos.size(1)
		num_neg = eval_edges_neg.size(1)
		eval_labels = num_pos * [1]
		eval_labels.extend(num_neg * [0])
		eval_labels = np.array(eval_labels)

		aucs = []
		for src_idxs in DataLoader(range(eval_edges.size(1)), 2 * EVAL_BATCH_SIZE, shuffle = True):
			scores = sg_model(eval_edges[0][src_idxs], eval_edges[1][src_idxs]).cpu()
			aucs.append(roc_auc_score(
				y_score = scores, 
				y_true = eval_labels[src_idxs.cpu()]))
		avg_auc = np.mean(aucs)
		return "AUC ROC", avg_auc

	@staticmethod
	@torch.no_grad()
	def test_mrr(sg_model, dataset, test_set):
		def _get_negative_scores(sg_model, dataset, test_set):
			'''
			negatives_for_hits: tensor of shape (n, K) containing the indices of random dst nodes
			'''
			scores = []
			eval_edges = dataset.get_eval_data(test_set)
			for src_idxs in DataLoader(range(eval_edges.size(1)), EVAL_BATCH_SIZE, shuffle = False):
				negatives_for_mrr = dataset.get_mrr_negatives(src_idxs, test_set)
				K = negatives_for_mrr.size(1)

				src = eval_edges[0][src_idxs].repeat_interleave(K)
				dst = negatives_for_mrr.view(-1)
				scores.append(
					sg_model(src, dst)\
					.view(len(src_idxs), -1)
				)
			return torch.cat(scores, dim=0)

		def _get_mrr(y_pos, y_neg):
			y_pos = y_pos.view(-1, 1)
			# optimistic rank: "how many negatives have a larger score than the positive?"
			# ~> the positive is ranked first among those with equal score
			optimistic_rank = (y_neg > y_pos).sum(dim=1)
			# pessimistic rank: "how many negatives have at least the positive score?"
			# ~> the positive is ranked last among those with equal score
			pessimistic_rank = (y_neg >= y_pos).sum(dim=1)
			ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
			mrr_list = 1./ranking_list.to(torch.float)
			return mrr_list.reshape((1, -1))
			# return mrr_list.mean()

		sg_model.eval()

		eval_edges = dataset.get_eval_data(test_set)
		y_neg = _get_negative_scores(sg_model, dataset, test_set)
		total_mrr = 0.
		all_mrr = []
		for eval_batch in DataLoader(range(eval_edges.size(1)), EVAL_BATCH_SIZE, shuffle = False):
			eval_batch_edges = eval_edges[:, eval_batch]
			y_pos = sg_model(eval_batch_edges[0], eval_batch_edges[1])
			# total_mrr += len(eval_batch) * _get_mrr(y_pos, y_neg[eval_batch])
			all_mrr.append(_get_mrr(y_pos, y_neg[eval_batch]))
		# avg_mrr = total_mrr / eval_edges.size(1)
		# print(f'MRR: {avg_mrr:.4f}')
		# return "MRR", avg_mrr
		all_mrr = torch.cat(all_mrr, dim=1)
		return "MRR", all_mrr

	@staticmethod
	@torch.no_grad()
	def test_hits(sg_model, K, dataset, test_set):
		def _get_negative_scores(sg_model, negatives_for_hits):
			'''
			negatives_for_hits: tensor of shape (2, num_negatives) containing indices of nodes
			'''
			return sg_model(negatives_for_hits[0], negatives_for_hits[1])

		def _get_hits(y_pos, y_neg, K):
			threshold = torch.topk(y_neg, K).values[-1]
			return (y_pos > threshold).float().reshape((1, -1))
			# return torch.sum(y_pos > threshold) / len(y_pos)

		sg_model.eval()
		eval_edges = dataset.get_eval_data(test_set)
		negatives_for_hits = dataset.get_hits_negatives(test_set)
		device = negatives_for_hits.device

		y_neg = _get_negative_scores(sg_model, negatives_for_hits)
		total_hits = 0.
		all_hits = []
		for eval_batch in DataLoader(range(eval_edges.size(1)), EVAL_BATCH_SIZE, shuffle = False):
			# print(test_batch)
			eval_batch_edges = eval_edges[:, eval_batch]
			y_pos = sg_model(
				eval_batch_edges[0], 
				eval_batch_edges[1]
			)
			hits_batch = _get_hits(y_pos, y_neg, K)

			# total_hits += hits_batch * eval_batch.size(0)
			all_hits.append(hits_batch)

		all_hits = torch.cat(all_hits, dim=1)
		# avg_hits = total_hits / eval_edges.size(1)
		# print(f'Hits@{K}: {avg_hits:.4f}')
		# return f'Hits@{K}', avg_hits
		return f'Hits@{K}', all_hits
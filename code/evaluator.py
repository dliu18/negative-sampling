import torch
from torch.utils.data import DataLoader
from dataloader import BasicDataset
from model import BasicModel

EVAL_BATCH_SIZE = 128

class Evaluator:
	@staticmethod
	@torch.no_grad()
	def test_mrr(sg_model, dataset):
		def _get_negative_scores(sg_model, test_edges, negatives_for_mrr):
			'''
			negatives_for_hits: tensor of shape (n, K) containing the indices of random dst nodes
			'''
			scores = []
			K = negatives_for_mrr.size(1)
			for src_idxs in DataLoader(range(negatives_for_mrr.size(0)), EVAL_BATCH_SIZE, shuffle = False):
				src = test_edges[0][src_idxs].repeat_interleave(K)
				dst = negatives_for_mrr[src_idxs].view(-1)
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
			return mrr_list.mean()

		sg_model.eval()
		test_edges = dataset.get_test_data()
		negatives_for_mrr = dataset.get_mrr_negatives()

		y_neg = _get_negative_scores(sg_model, test_edges, negatives_for_mrr)
		total_mrr = 0.
		for test_batch in DataLoader(range(dataset.n_test_edges), EVAL_BATCH_SIZE, shuffle = False):
			test_batch_edges = test_edges[:, test_batch]
			y_pos = sg_model(test_batch_edges[0], test_batch_edges[1])
			total_mrr += len(test_batch) * _get_mrr(y_pos, y_neg[test_batch])
		avg_mrr = total_mrr / test_edges.size(1)
		print(f'MRR: {avg_mrr:.4f}')
		return "MRR", avg_mrr

	@staticmethod
	@torch.no_grad()
	def test_hits(sg_model, dataset):
		def _get_negative_scores(sg_model, negatives_for_hits):
			'''
			negatives_for_hits: tensor of shape (2, num_negatives) containing indices of nodes
			'''
			return sg_model(negatives_for_hits[0], negatives_for_hits[1])

		def _get_hits(y_pos, y_neg, K):
			threshold = torch.topk(y_neg, K).values[-1]
			return torch.sum(y_pos > threshold) / len(y_pos)

		sg_model.eval()
		test_edges = dataset.get_test_data()
		negatives_for_hits = dataset.get_hits_negatives()
		device = negatives_for_hits.device
		K = 50

		y_neg = _get_negative_scores(sg_model, negatives_for_hits)
		total_hits = 0.
		for test_batch in DataLoader(range(test_edges.size(1)), EVAL_BATCH_SIZE, shuffle = False):
			# print(test_batch)
			test_batch_edges = test_edges[:, test_batch]
			y_pos = sg_model(
				test_batch_edges[0], 
				test_batch_edges[1]
			)
			hits_batch = _get_hits(y_pos, y_neg, K)

			total_hits += hits_batch * test_batch.size(0)

		avg_hits = total_hits / test_edges.size(1)
		print(f'Hits@{K}: {avg_hits:.4f}')
		return f'Hits@{K}', avg_hits
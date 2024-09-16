import torch
from torch.utils.data import DataLoader

from dataloader import dataset
from model import BasicModel

EVAL_BATCH_SIZE = 128

class Evaluator:
	@staticmethod
	@torch.no_grad()
	def test_mrr(sg_model, dataset):
		def _get_negative_scores(sg_model, negatives_for_mrr):
			'''
			negatives_for_hits: tensor of shape (n, K) containing the indices of random dst nodes
			'''
			scores = []
			K = negatives_for_mrr.size(1)
			for src_batch in DataLoader(range(negatives_for_mrr.size(0)), EVAL_BATCH_SIZE, shuffle = False):
				dst = negatives_for_mrr[src_batch].view(-1)
				src_batch = src_batch.repeat_interleave(K)
				src_emb = sg_model(src_batch)
				dst_emb = sg_model(dst)
				scores.append(torch.mul(src_emb, dst_emb).sum(dim = 1).view(int(len(src_batch) / K), -1))
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

		y_neg = _get_negative_scores(sg_model, negatives_for_mrr, K)
		total_mrr = 0.
		for test_batch in DataLoader(range(dataset.n_test_edges), EVAL_BATCH_SIZE, shuffle = False):
			test_batch_edges = test_edges[:, test_batch]
			src_emb = sg_model(test_batch_edges[0])
			dst_emb = sg_model(test_batch_edges[1])
			y_pos = torch.mul(src_emb, dst_emb).sum(dim = 1)
			total_mrr += len(test_batch) * _get_mrr(y_pos, y_neg[test_batch_edges[0]])
		avg_mrr = total_mrr / test_data.num_edges
		print(f'MRR: {avg_mrr:.4f}')
		return "MRR", avg_mrr

	@staticmethod
	@torch.no_grad()
	def test_hits(sg_model, dataset):
		def _get_negative_scores(sg_model, negatives_for_hits):
			'''
			negatives_for_hits: tensor of shape (2, num_negatives) containing indices of nodes
			'''
			src_emb = sg_model(negatives_for_hits[0])
			dst_emb = sg_model(negatives_for_hits[1])
			return torch.mul(src_emb, dst_emb).sum(dim = 1)

		def _get_hits(y_pos, y_neg, K):
			threshold = torch.topk(y_neg, K).values[-1]
			return torch.sum(y_pos > threshold) / len(y_pos)

		sg_model.eval()
		test_edges = dataset.get_test_data()
		negatives_for_hits = dataset.get_hits_negatives()
		K = negatives_for_hits.size(1)

		y_neg = _get_negative_scores(sg_model, negatives_for_hits)
		total_hits = 0.
		for test_batch in DataLoader(range(test_data.num_edges), 128, shuffle = False):
			# print(test_batch)
			test_batch_edges = test_edges[:, test_batch]
			src_emb = sg_model(test_batch_edges[0])
			dst_emb = sg_model(test_batch_edges[1])
			y_pos = torch.mul(src_emb, dst_emb).sum(dim = 1)
			hits_batch = _get_hits(y_pos, y_neg, K)

			total_hits += hits_batch * test_batch.size(0)

		avg_hits = total_hits / test_data.num_edges
		print(f'Hits@{K}: {avg_hits:.4f}')
		return f'Hits@{K}', avg_hits
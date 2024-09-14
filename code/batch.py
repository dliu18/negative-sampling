import torch
from torch.utils.data import DataLoader

from torch_geometric.nn import Node2Vec
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.sampler import BaseSampler, EdgeSamplerInput, NegativeSampling, SamplerOutput
from tqdm import tqdm

from ogb.linkproppred import Evaluator
from ogb.linkproppred import PygLinkPropPredDataset

@torch.no_grad()
def test_mrr(model, negatives_for_mrr, test_data, K):
	def _get_negative_scores(model, negatives_for_mrr, K):
		'''
		negatives_for_hits: tensor of shape (n, K) containing the indices of random dst nodes
		'''
		# may need to process in batches if there are memory issues
		scores = []
		batch_size = 10
		for src_batch in DataLoader(range(negatives_for_mrr.size(0)), batch_size, shuffle = False):
			dst = negatives_for_mrr[src_batch].view(-1)
			src_batch = src_batch.repeat_interleave(K)
			src_emb = model(src_batch)
			dst_emb = model(dst)
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

	model.eval()
	test_edges = test_data.edge_index
	y_neg = _get_negative_scores(model, negatives_for_mrr, K)
	total_mrr = 0.
	for test_batch in DataLoader(range(test_data.num_edges), 10, shuffle = False):
		test_batch_edges = test_edges[:, test_batch]
		src_emb = model(test_batch_edges[0])
		dst_emb = model(test_batch_edges[1])
		y_pos = torch.mul(src_emb, dst_emb).sum(dim = 1)
		total_mrr += len(test_batch) * _get_mrr(y_pos, y_neg[test_batch_edges[0]])
	avg_mrr = total_mrr / test_data.num_nodes
	print(f'MRR: {avg_mrr:.4f}')

@torch.no_grad()
def test_hits(model, negatives_for_hits, test_data, K):
	def _get_negative_scores(model, negatives_for_hits):
		'''
		negatives_for_hits: tensor of shape (2, num_negatives) containing indices of nodes
		'''
		src_emb = model(negatives_for_hits[0])
		dst_emb = model(negatives_for_hits[1])
		return torch.mul(src_emb, dst_emb).sum(dim = 1)

	def _get_hits(y_pos, y_neg, K):
		threshold = torch.topk(y_neg, K).values[-1]
		return torch.sum(y_pos > threshold) / len(y_pos)

	model.eval()
	test_edges = test_data.edge_index
	y_neg = _get_negative_scores(model, negatives_for_hits)
	total_hits = 0.
	for test_batch in DataLoader(range(test_data.num_edges), 128, shuffle = False):
		# print(test_batch)
		test_batch_edges = test_edges[:, test_batch]
		src_emb = model(test_batch_edges[0])
		dst_emb = model(test_batch_edges[1])
		y_pos = torch.mul(src_emb, dst_emb).sum(dim = 1)
		hits_batch = _get_hits(y_pos, y_neg, K)

		total_hits += hits_batch * test_batch.size(0)

	avg_hits = total_hits / test_data.num_edges
	print(f'Hits@{K}: {avg_hits:.4f}')
	return avg_hits

def train(model,
	loader,
	optimizer
):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model.train()
	total_loss = 0
	for pos_rw, neg_rw in tqdm(loader):
		# print("Positive RW shape: ", pos_rw.shape, " Negative RW shape: ", neg_rw.shape)
		optimizer.zero_grad()
		loss = model.loss(pos_rw.to(device), neg_rw.to(device))
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	return total_loss / len(loader)

if __name__ == "__main__":
	# dataset = Planetoid(
	# 	root = "../data/",
	# 	name = "CiteSeer",
	# )
	dataset = PygLinkPropPredDataset(name='ogbl-collab')
	data = dataset[0]
	print("loaded the dataset")

	split = RandomLinkSplit(is_undirected=data.is_undirected(),
		num_val = 0.0,
		num_test = 0.2,
		add_negative_train_samples = False
	)

	train_data, _, test_data = split(data)
	print("split the dataset")

	model = Node2Vec(to_undirected(train_data.edge_index, train_data.num_nodes), 
				embedding_dim=128, 
				 walk_length=40,                        # lenght of rw
				 context_size=20, walks_per_node=10,
				 num_negative_samples=1, 
				 p=1, q=1,                             # bias parameters
				 sparse=True).to('cuda')

	loader = model.loader(batch_size=128, shuffle=True, num_workers=1)
	optimizer = torch.optim.SparseAdam(list(model.parameters()), lr = 0.01)

	K_hits = 50
	negatives_for_hits = torch.randint(high=data.num_nodes, size=(2, 100000))

	K_mrr = 1000
	negatives_for_mrr = torch.randint(high=data.num_nodes, size=(data.num_nodes, K_mrr))

	for epoch in range(1, 100):
		if epoch % 1 == 0:
			avg_hits = test_hits(model, negatives_for_hits, test_data, K_hits)
			avg_mrr = test_mrr(model, negatives_for_mrr, test_data, K_mrr)
		loss = train(model, loader, optimizer)
		print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')


# questions: 
# where are the negative samples stored in RandomLinkSplit
# how do the split masks work. Are they pre-defined in the dataset?
# reference this for loading the ogb datasets
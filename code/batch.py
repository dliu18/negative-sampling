import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import RandomLinkSplit
from tqdm import tqdm

def train(model,
	loader,
	optimizer
):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model.train()
	total_loss = 0
	for pos_rw, neg_rw in tqdm(loader):
		print("Positive RW shape: ", pos_rw.shape, " Negative RW shape: ", neg_rw.shape)
		optimizer.zero_grad()
		loss = model.loss(pos_rw.to(device), neg_rw.to(device))
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	return total_loss / len(loader)

if __name__ == "__main__":
	dataset = Planetoid(
		root = "../data/",
		name = "CiteSeer",
	)
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
				 walk_length=20,                        # lenght of rw
				 context_size=10, walks_per_node=20,
				 num_negative_samples=1, 
				 p=200, q=1,                             # bias parameters
				 sparse=True).to('cuda')

	loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
	optimizer = torch.optim.SparseAdam(list(model.parameters()), lr = 0.01)

	for epoch in range(1, 100):
		loss = train(model, loader, optimizer)
		print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')


# questions: 
# where are the negative samples stored in RandomLinkSplit
# how do the split masks work. Are they pre-defined in the dataset?
# reference this for loading the ogb datasets
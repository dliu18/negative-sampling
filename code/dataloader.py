import os
from os.path import join
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import world
from time import time

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import Node2Vec

from ogb.linkproppred import PygLinkPropPredDataset

NUM_HITS_NEGATIVES = int(1e5)
NUM_MRR_NEGATIVES = int(1e3)

class BasicDataset(Dataset):
    def __init__(self, name, seed=0):
        self.name = name
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("init dataset: ", self.name, " device: ", self.device)
    
    @property
    def dataset_name(self):
        return self.name

    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def n_train_edges(self):
        raise NotImplementedError

    @property
    def n_test_edges(self):
        raise NotImplementedError

    def get_train_loader_rw(self, batch_size, sample_negatives):
        # batches of edges
        raise NotImplementedError

    def get_train_loader_edges(self, batch_size, sample_negatives):
        '''
        Data loader that returns batches of edges (positives) and uniform random target nodes (negatives)
        '''
        raise NotImplementedError

    def get_test_data(self):
        '''
        Returns the full test edge set in COO format. Values are node indices. Shape should be (2, num_test)
        '''
        raise NotImplementedError

    def get_hits_negatives(self): 
        '''
        Returns the negatives used to compute Hits@K. The negatives are shared by all positive edges and there are a
        predetermined number of negatives. Output is of shape (2, global_num_negative)
        '''
        raise NotImplementedError

    def get_mrr_negatives(self):
        '''
        Returns the negatives used to compute MRR. Each test edge has a slate of "num_neg" negatives to compare against.
        Output is of shape (num_test, num_neg)
        '''
        raise NotImplementedError

class SmallBenchmark(BasicDataset):
    '''
    Class for the Cora, CiteSeer, and PubMed datasets.
    '''
    def __init__(self, name, seed):
        super().__init__(name, seed)

        assert name in ["Cora", "CiteSeer", "PubMed"]
        dataset = Planetoid(
            root = "dataset/",
            name = name,
        )
        data = dataset[0]
        self.full_data = data

        split = RandomLinkSplit(is_undirected=data.is_undirected(),
            num_val = 0.0,
            num_test = 0.2,
            add_negative_train_samples = False
        )

        train_data, _, test_data = split(data)

        self.train_data = train_data
        self.test_data = test_data

    @property
    def n_users(self):
        return self.full_data.num_nodes

    @property
    def n_train_edges(self):
        return self.train_data.num_edges

    @property
    def n_test_edges(self):
        return self.test_data.num_edges

    def get_train_loader_rw(self, batch_size, sample_negatives):
        model = Node2Vec(to_undirected(self.train_data.edge_index, self.full_data.num_nodes), 
                    embedding_dim=128, # set as a placeholder but the embeddings in here are not used
                     walk_length=20,
                     context_size=10,
                     walks_per_node = 1,
                     num_negative_samples= 1 if sample_negatives else 0
                ).to(self.device)

        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=1)
        return loader

    def _sample_edges(self, batch):
        pos = self.train_edges[:, batch].to(self.device)
        neg = pos.clone()

        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed)
        neg[1] = torch.randint(high=self.n_users, generator=generator, size=neg.size(1)).to(self.device)
        return pos.t(), neg.t()

    def get_train_loader_edges(self, batch_size, sample_negatives):
        '''
        Data loader that returns batches of edges (positives) and uniform random target nodes (negatives)
        '''
        return DataLoader(range(self.n_train_edges), collate_fn=self._sample_edges)
    
    def get_test_data(self):
        '''
        Returns the full test edge set in COO format. Values are node indices. Shape should be (2, num_test)
        '''
        return self.test_data.edge_index.to(self.device)

    def get_hits_negatives(self):
        '''
        Returns the negatives used to compute Hits@K. The negatives are shared by all positive edges and there are a
        predetermined number of negatives. Output is of shape (2, global_num_negative)
        '''
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed)
        return torch.randint(high=self.n_users, generator=generator, size=(2, NUM_HITS_NEGATIVES)).to(self.device)

    def get_mrr_negatives(self):
        '''
        Returns the negatives used to compute MRR. Each test edge has a slate of "num_neg" negatives to compare against.
        Output is of shape (num_test, num_neg)
        '''
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed)
        return torch.randint(high=self.n_users, generator=generator, size=(self.n_test_edges, NUM_MRR_NEGATIVES)).to(self.device)

class OGBBenchmark(BasicDataset):
    '''
    Class for the OGB link prediction datsets
    '''
    def __init__(self, name, seed):
        super().__init__(name, seed)

        assert name in ["ogbl-collab", "ogbl-ppa", "ogbl-citation2"]
        dataset = PygLinkPropPredDataset(name=name)
        data = dataset[0]
        self.full_data = data

        self.split_edge = dataset.get_edge_split()
        self.train_edges = self.split_edge['train']['edge'].t().to(device=self.device)
        self.test_edges = self.split_edge['test']['edge'].t().to(device=self.device)

    @property
    def n_users(self):
        return self.full_data.num_nodes

    @property
    def n_train_edges(self):
        return self.train_edges.size(1)

    @property
    def n_test_edges(self):
        return self.test_edges.size(1)

    def get_train_loader_rw(self, batch_size, sample_negatives):
        model = Node2Vec(to_undirected(self.train_edges, self.full_data.num_nodes), 
                    embedding_dim=128, # set as a placeholder but the embeddings in here are not used
                     walk_length=40,
                     context_size=20,
                     walks_per_node = 10,
                     num_negative_samples= 1 if sample_negatives else 0
                ).to(self.device)

        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=1)
        return loader

    def get_test_data(self):
        '''
        Returns the full test edge set in COO format. Values are node indices. Shape should be (2, num_test)
        '''
        return self.test_edges.to(self.device)

    def get_hits_negatives(self):
        '''
        Returns the negatives used to compute Hits@K. The negatives are shared by all positive edges and there are a
        predetermined number of negatives. Output is of shape (2, global_num_negative)
        '''
        if "edge_neg" in self.split_edge["test"]:
            return self.split_edge["test"]["edge_neg"].t().to(self.device)
        else:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(self.seed)
            return torch.randint(high=self.sg_model, generator=generator, size=(2, NUM_HITS_NEGATIVES)).to(self.device)


    def get_mrr_negatives(self):
        '''
        Returns the negatives used to compute MRR. Each test edge has a slate of "num_neg" negatives to compare against.
        Output is of shape (num_test, num_neg)
        '''
        if "target_node_neg" in self.split_edge["test"]:
            return self.split_edge["test"]["target_node_neg"].to(self.device)
        else:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(self.seed)
            return torch.randint(high=self.n_users, generator=generator, size=(self.n_test_edges, NUM_MRR_NEGATIVES)).to(self.device)


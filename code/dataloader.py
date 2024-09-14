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

class BasicDataset(Dataset):
    def __init__(self, name):
        self.name = name
        print("init dataset")
    
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

    def get_train_loader(self, batch_size, sample_negatives):
        # batches of edges
        raise NotImplementedError

    def get_test_loader(self, batch_size, sample_negatives):
        # batches of nodes
        raise NoteImplementedError

class SmallBenchmark(BasicDataset):
    def __init__(self, name):
        super().__init__(name)

        assert name in ["Cora", "CiteSeer", "PubMed"]
        dataset = Planetoid(
            root = "../data/",
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
        return self.train_data.num_nodes

    @property
    def n_train_edges(self):
        return self.train_data.num_edges

    @property
    def n_test_edges(self):
        return self.test_data.num_edges

    def get_train_loader(self, batch_size, sample_negatives):
        model = Node2Vec(to_undirected(self.train_data.edge_index, self.train_data.num_nodes), 
                    embedding_dim=128, # set as a placeholder but the embeddings in here are not used
                     walk_length=20,
                     context_size=10,
                     walks_per_node = 1,
                     num_negative_samples= 1 if sample_negatives else 0
                ).to('cuda')

        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=1)
        return loader
    

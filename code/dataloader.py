import os
from os.path import join
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import world
from time import time

from torch_geometric.datasets import Planetoid, 
from torch_geometric.utils import to_undirected, degree
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
    def n_train_edges(self):
        raise NotImplementedError

    @property
    def n_valid_edges(self):
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

    def get_valid_data(self):
        '''
        Returns the full validation edge set in COO format. Values are node indices. Shape should be (2, num_valid)
        '''
        raise NotImplementedError

    def get_test_data(self):
        '''
        Returns the full test edge set in COO format. Values are node indices. Shape should be (2, num_test)
        '''
        raise NotImplementedError

    def get_eval_data(self, test_set):
        if test_set == "test":
            return self.get_test_data()
        elif test_set == "valid":
            return self.get_valid_data()
        else:
            raise NotImplementedError

    def get_sg_negatives(self, shape, alpha=0.75):
        raise NotImplementedError
            
    def get_hits_negatives(self, test_set="test"): 
        '''
        Returns the negatives used to compute Hits@K. The negatives are shared by all positive edges and there are a
        predetermined number of negatives. Output is of shape (2, global_num_negative)
        test_set is either "test" or "valid"
        '''
        raise NotImplementedError

    def get_mrr_negatives(self, edge_idxs, test_set="test"):
        '''
        Returns the negatives used to compute MRR. Each test edge has a slate of "num_neg" negatives to compare against.
        edge_idxs contains the indices of the test set that are being requested. This avoids generating the entire set of 
        negatives for all test edges at once to save memory.
        Output is of shape (len(edge_idxs), num_neg)
        test_set is either "test" or "valid"
        '''
        raise NotImplementedError

class SmallBenchmark(BasicDataset):
    '''
    Class for the Cora, CiteSeer, and PubMed datasets.
    '''
    def __init__(self, name, seed):
        super().__init__(name, seed)

        assert name in ["Cora", "CiteSeer", "PubMed"] or "SBM" in name

        dataset = None
        if name in ["Cora", "CiteSeer", "PubMed"]:
            dataset = Planetoid(
                root = "dataset/",
                name = name,
            )
        elif "SBM" in name:
            return

        data = dataset[0]
        self.full_data = data

        split = RandomLinkSplit(is_undirected=data.is_undirected(),
            num_val = 0.1,
            num_test = 0.2,
            add_negative_train_samples = False
        )

        train_data, valid_data, test_data = split(data)

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.degrees = degree(train_data.edge_index[0]).to(self.device)

        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed)
        self.generator = generator

    @property
    def n_users(self):
        return self.full_data.num_nodes

    @property
    def n_train_edges(self):
        return self.train_data.num_edges

    @property
    def n_valid_edges(self):
        return self.valid_data.num_edges

    @property
    def n_test_edges(self):
        return self.test_data.num_edges

    def get_train_loader_rw(self, batch_size, sample_negatives):
        model = Node2Vec(self.train_data.edge_index, 
                    embedding_dim=128, # set as a placeholder but the embeddings in here are not used
                     walk_length=60,
                     context_size=20,
                     walks_per_node = 10,
                     num_negative_samples= 1 if sample_negatives else 0
                ).to(self.device)

        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=1)
        return loader

    def _sample_edges(self, batch):
        pos = self.train_data.edge_index[:, batch].to(self.device)
        neg = pos.clone()

        neg[1] = torch.randint(high=self.n_users, generator=self.generator, size=(neg.size(1),)).to(self.device)
        return pos.t(), neg.t()

    def get_train_loader_edges(self, batch_size, sample_negatives):
        '''
        Data loader that returns batches of edges (positives) and uniform random target nodes (negatives)
        '''
        return DataLoader(range(self.n_train_edges), collate_fn=self._sample_edges, batch_size=batch_size)
    
    def get_valid_data(self):
        '''
        Returns the full validation edge set in COO format. Values are node indices. Shape should be (2, num_valid)
        '''
        return self.valid_data.edge_index.to(self.device)

    def get_test_data(self):
        '''
        Returns the full test edge set in COO format. Values are node indices. Shape should be (2, num_test)
        '''
        return self.test_data.edge_index.to(self.device)

    def get_sg_negatives(self, shape, alpha=0.75):
        weights = self.degree.pow(alpha)
        num_samples = 1
        for dim in shape:
            num_samples *= dim
        negatives = torch.multinomial(weights, num_samples = negatives, replacement=True).to(self.device)
        return negatives.reshape(shape)

    def get_hits_negatives(self, test_set="test"):
        '''
        Returns the negatives used to compute Hits@K. The negatives are shared by all positive edges and there are a
        predetermined number of negatives. Output is of shape (2, global_num_negative)
        '''
        return torch.randint(high=self.n_users, generator=self.generator, size=(2, NUM_HITS_NEGATIVES)).to(self.device)

    def get_mrr_negatives(self, edge_idxs, test_set="test"):
        '''
        Returns the negatives used to compute MRR. Each test edge has a slate of "num_neg" negatives to compare against.
        Output is of shape (num_test, num_neg)
        '''
        return torch.randint(high=self.n_users, generator=self.generator, size=(len(edge_idxs), NUM_MRR_NEGATIVES)).to(self.device)

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

        if 'edge' in self.split_edge['train']:
            self.train_edges = self.split_edge['train']['edge'].t().to(device=self.device)
            self.valid_edges = self.split_edge['valid']['edge'].t().to(device=self.device)
            self.test_edges = self.split_edge['test']['edge'].t().to(device=self.device)
        elif 'source_node' in self.split_edge['train']:
            self.train_edges = torch.cat([
                self.split_edge['train']['source_node'].reshape(1, -1),
                self.split_edge['train']['target_node'].reshape(1, -1)
                ], dim=0).to(device=self.device)

            self.valid_edges = torch.cat([
                self.split_edge['valid']['source_node'].reshape(1, -1),
                self.split_edge['valid']['target_node'].reshape(1, -1)
                ], dim=0).to(device=self.device)

            self.test_edges = torch.cat([
                self.split_edge['test']['source_node'].reshape(1, -1),
                self.split_edge['test']['target_node'].reshape(1, -1)
                ], dim=0).to(device=self.device)
        else:
            raise NotImplementedError("OGB dataset does not have correct schema: ",
                self.split_edge['train'].keys())

        self.degrees = degree(self.train_edges.reshape(-1)).to(self.device)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed)
        self.generator = generator

    @property
    def n_users(self):
        return self.full_data.num_nodes

    @property
    def n_train_edges(self):
        return self.train_edges.size(1)

    @property
    def n_valid_edges(self):
        return self.valid_edges.size(1)

    @property
    def n_test_edges(self):
        return self.test_edges.size(1)

    def get_train_loader_rw(self, batch_size, sample_negatives):
        edges = self.train_edges
        #if the graph is undirected we need to add in the bidirectional edges since OGB does not include them in split_edge
        if self.full_data.is_undirected();
            edges = to_undirected(self.train_edges, self.full_data.num_nodes)
        model = Node2Vec(edges, 
                    embedding_dim=128, # set as a placeholder but the embeddings in here are not used
                     walk_length=40,
                     context_size=20,
                     walks_per_node = 10,
                     num_negative_samples= 1 if sample_negatives else 0
                ).to(self.device)

        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=1)
        return loader

    def _sample_edges(self, batch):
        pos = self.train_edges[:, batch].to(self.device)
        neg = pos.clone()

        neg[1] = torch.randint(high=self.n_users, generator=self.generator, size=(neg.size(1),)).to(self.device)
        return pos.t(), neg.t()

    def get_train_loader_edges(self, batch_size, sample_negatives):
        '''
        Data loader that returns batches of edges (positives) and uniform random target nodes (negatives)
        '''
        return DataLoader(range(self.n_train_edges), collate_fn=self._sample_edges, batch_size=batch_size)
    
    def get_valid_data(self):
        '''
        Returns the full validation edge set in COO format. Values are node indices. Shape should be (2, num_valid)
        '''
        return self.valid_edges.to(self.device)

    def get_test_data(self):
        '''
        Returns the full test edge set in COO format. Values are node indices. Shape should be (2, num_test)
        '''
        return self.test_edges.to(self.device)

    def get_sg_negatives(self, shape, alpha=0.75):
        weights = self.degree.pow(alpha)
        num_samples = 1
        for dim in shape:
            num_samples *= dim
        negatives = torch.multinomial(weights, num_samples = negatives, replacement=True).to(self.device)
        return negatives.reshape(shape)


    def get_hits_negatives(self, test_set="test"):
        '''
        Returns the negatives used to compute Hits@K. The negatives are shared by all positive edges and there are a
        predetermined number of negatives. Output is of shape (2, global_num_negative)
        '''
        if "edge_neg" in self.split_edge[test_set]:
            return self.split_edge[test_set]["edge_neg"].t().to(self.device)
        else:
            return torch.randint(high=self.n_users, generator=self.generator, size=(2, NUM_HITS_NEGATIVES)).to(self.device)


    def get_mrr_negatives(self, edge_idxs, test_set="test"):
        '''
        Returns the negatives used to compute MRR. Each test edge has a slate of "num_neg" negatives to compare against.
        Output is of shape (num_test, num_neg)
        '''
        if "target_node_neg" in self.split_edge[test_set]:
            return self.split_edge[test_set]["target_node_neg"][edge_idxs].to(self.device)
        else:
            return torch.randint(high=self.n_users, generator=self.generator, size=(len(edge_idxs), NUM_MRR_NEGATIVES)).to(self.device)


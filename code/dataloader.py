import os
from os.path import join
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import world
from time import time

from torch_geometric.datasets import Planetoid, StochasticBlockModelDataset, SNAPDataset
from torch_geometric.utils import to_undirected, to_networkx, degree
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import Node2Vec

from ogb.linkproppred import PygLinkPropPredDataset

import networkx as nx
from networkx.algorithms import cluster 

NUM_HITS_NEGATIVES = int(1e5)
NUM_MRR_NEGATIVES = int(1e3)

class BasicDataset(Dataset):
    def __init__(self, name, test_set, seed=0):
        self.name = name
        self.seed = seed
        self.test_set = test_set
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

    def get_degrees(self):
        raise NotImplementedError

    def get_clustering_coefs(self):
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

    def get_roc_negatives(self):
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
    def __init__(self, name, test_set="test", test_set_frac=0.2, seed=2020):
        super().__init__(name, test_set, seed)

        assert name in ["Cora", "CiteSeer", "PubMed", "ego-facebook", "soc-ca-astroph"] or "SBM" in name

        dataset = None
        if name in ["Cora", "CiteSeer", "PubMed"]:
            dataset = Planetoid(
                root = "dataset/",
                name = name,
            )
        elif "SBM" in name:
            _, p, q = name.split("-")
            blocks = 2
            p = float(p)
            q = float(q)
            B = q * torch.ones(blocks, blocks) + (p - q) * torch.eye(blocks)
            dataset = StochasticBlockModelDataset(
                root = "/work/radlab/David/pyg",
                block_sizes = [1000] * blocks,
                edge_probs = B,
                is_undirected = True,
                force_reload = True) #PyG stores edge prob rounded to 1 decimal point in fileanme so must reload
        elif name in ["ego-facebook", "soc-ca-astroph"]:
            dataset = SNAPDataset(
                root = "dataset/",
                name = name
            )

        data = dataset[0]
        self.full_data = data

        split = RandomLinkSplit(is_undirected=data.is_undirected(),
            num_val = 0.1,
            num_test = test_set_frac,
            add_negative_train_samples = False
        )

        train_data, valid_data, test_data = split(data)

        self.train_edges = train_data.edge_label_index[:, train_data.edge_label == 1]
        self.valid_edges = valid_data.edge_label_index[:, valid_data.edge_label == 1]
        self.valid_edges_neg = valid_data.edge_label_index[:, valid_data.edge_label == 0]
        self.test_edges = test_data.edge_label_index[:, test_data.edge_label == 1]
        self.test_edges_neg = test_data.edge_label_index[:, test_data.edge_label == 0]

        if data.is_undirected():
            self.train_edges = to_undirected(self.train_edges)
            self.valid_edges = to_undirected(self.valid_edges)
            self.valid_edges_neg = to_undirected(self.valid_edges_neg)
            self.test_edges = to_undirected(self.test_edges)
            self.test_edges_neg = to_undirected(self.test_edges_neg)

        reconstructed =  torch.cat([self.train_edges, self.valid_edges, self.test_edges], dim=1)
        assert reconstructed.shape == data.edge_index.shape
        assert (reconstructed.sum(dim=1) == data.edge_index.sum(dim=1)).all()

        if test_set == "test":
            self.train_edges = torch.cat([self.train_edges, self.valid_edges], dim=1)
        self.degrees = degree(self.train_edges[0], num_nodes=data.num_nodes).to(self.device)

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

    def get_degrees(self):
        return self.degrees

    def get_clustering_coefs(self):
        try:
            return np.load(f"dataset/cluster/{self.name}_cluster_coefs.npy")
        except:
            print("Starting conversion to networkx for: ", self.name)
            graph_nx = to_networkx(self.full_data, to_undirected = self.full_data.is_undirected())
            print("Finished conversion to networkx for: ", self.name)

            cluster_coefs_dict = cluster.clustering(graph_nx)
            cluster_coefs = np.zeros(self.n_users)
            for idx in range(self.n_users):
                cluster_coefs[idx] = cluster_coefs_dict[idx]
            np.save(f"dataset/cluster/{self.name}_cluster_coefs.npy", cluster_coefs)
            return cluster_coefs


    def get_train_loader_rw(self, batch_size, sample_negatives, p=1.0, q=1.0):
        model = Node2Vec(self.train_edges, 
                    embedding_dim=128, # set as a placeholder but the embeddings in here are not used
                     walk_length=60,
                     context_size=20,
                     walks_per_node = 10,
                     num_negative_samples= 1 if sample_negatives else 0,
                     p=p, q=q
                ).to(self.device)

        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=1)
        return loader

    def _sample_edges(self, batch):
        # pos = torch.randint(high=self.n_users, generator=self.generator, size=(2, len(batch))).to(self.device)
        # return pos.t(), None
        
        # pos = self.train_edges[:, batch].to(self.device)
        pos = self.train_edges[:, batch]
        # neg = pos.clone()

        # neg[1] = torch.randint(high=self.n_users, generator=self.generator, size=(neg.size(1),)).to(self.device)
        return pos.t(), None

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

    def get_roc_negatives(self, test_set="test"):
        assert test_set == "test" or test_set == "valid"
        if test_set == "test":
            return self.test_edges_neg.to(self.device)
        elif test_set == "valid":
            return self.valid_edges_neg.to(self.device)

    def get_sg_negatives(self, shape, alpha=0.75):
        weights = self.degrees.pow(alpha)
        num_samples = 1
        for dim in shape:
            num_samples *= dim
        negatives = torch.multinomial(weights, num_samples = num_samples, replacement=True).to(self.device)
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
    def __init__(self, name, test_set="test", seed=2020):
        super().__init__(name, test_set, seed)

        assert name in ["ogbl-collab", "ogbl-ppa", "ogbl-citation2", "ogbl-vessel"]
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

        if test_set == "test":
            self.train_edges = torch.cat([self.train_edges, self.valid_edges], dim=1)

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

    def get_degrees(self):
        return self.degrees

    def get_clustering_coefs(self):
        try:
            return np.load(f"dataset/cluster/{self.name}_cluster_coefs.npy")
        except:
            print("Starting conversion to networkx for: ", self.name)
            graph_nx = to_networkx(self.full_data, to_undirected = self.full_data.is_undirected())
            print("Finished conversion to networkx for: ", self.name)

            cluster_coefs_dict = cluster.clustering(graph_nx)
            cluster_coefs = np.zeros(self.n_users)
            for idx in range(self.n_users):
                cluster_coefs[idx] = cluster_coefs_dict[idx]
            np.save(f"dataset/cluster/{self.name}_cluster_coefs.npy", cluster_coefs)
            return cluster_coefs

    def get_train_loader_rw(self, batch_size, sample_negatives, p=1.0, q=1.0):
        edges = self.train_edges
        #if the graph is undirected we need to add in the bidirectional edges since OGB does not include them in split_edge
        if self.full_data.is_undirected():
            edges = to_undirected(self.train_edges, self.full_data.num_nodes)
        model = Node2Vec(edges, 
                    embedding_dim=128, # set as a placeholder but the embeddings in here are not used
                     walk_length=40,
                     context_size=20,
                     walks_per_node = 10,
                     num_negative_samples= 1 if sample_negatives else 0,
                     p=p, q=q
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

    def get_roc_negatives(self, test_set="test"):
        num_edges = self.n_test_edges if test_set == "test" else self.n_valid_edges
        return torch.randint(high=self.n_users, generator=self.generator, size=(2, num_edges)).to(self.device)

    def get_sg_negatives(self, shape, alpha=0.75):
        weights = self.degrees.pow(alpha)
        num_samples = 1
        for dim in shape:
            num_samples *= dim
        negatives = torch.multinomial(weights, num_samples = num_samples, replacement=True).to(self.device)
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

if __name__ == "__main__": 
    for name in ["Cora", "CiteSeer", "PubMed", "SBM-0.7-0.008"]:
        dataset = SmallBenchmark(name, seed = 2020)
        print(
            name, 
            "\n Nodes: ",
            dataset.n_users,
            "\n Train edges: ",
            dataset.n_train_edges,
            "\n Validation edges: ",
            dataset.n_valid_edges,
            "\n Test edges: ",
            dataset.n_test_edges,
            "\n Average Clustering Coefficient: ",
            np.mean(dataset.get_clustering_coefs()),
            "\n"
        )

    # for name in ["ogbl-collab", "ogbl-ppa", "ogbl-citation2"]:
    #     dataset = OGBBenchmark(name, seed = 2020)
    #     print(
    #         name, 
    #         "\n Nodes: ",
    #         dataset.n_users,
    #         "\n Train edges: ",
    #         dataset.n_train_edges,
    #         "\n Validation edges: ",
    #         dataset.n_valid_edges,
    #         "\n Test edges: ",
    #         dataset.n_test_edges,
    #         "\n Average Clustering Coefficient: ",
    #         np.mean(dataset.get_clustering_coefs()),
    #         "\n"
    #     )

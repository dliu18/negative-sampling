import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import world
import math

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def sg_positive_loss(self, source, target):
        raise NotImplementedError

    def sg_negative_loss(self, source, target):
        raise NotImplementedError

    def dimension_reg(self):
        raise NotImplementedError
    
class SGModel(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(SGModel, self).__init__()
        self.num_users  = dataset.n_users
        self.latent_dim = config['latent_dim_rec']
        self.lam = config["lambda"]
        self.alpha = config["alpha"]
        self.degrees = dataset.get_degrees()
        self.device = world.device
        self.eps = 1e-15
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = nn.Embedding(
            num_embeddings=self.num_users, 
            embedding_dim=self.latent_dim,
            device=self.device)

        # TODO: define the classifier 
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 64),  # Input: concatenated embeddings
            nn.ReLU(),
            nn.Linear(64, 1),  # Output: binary classification
            nn.Sigmoid()  # Sigmoid for binary output
        )

        # nn.init.uniform_(self.embedding_user.weight, 
        #     a=-math.sqrt(self.latent_dim), 
        #     b=math.sqrt(self.latent_dim))        

    def sg_positive_loss(self, source, target):
        u_emb = self.embedding_user(source.long())
        v_emb = self.embedding_user(target.long())
        dot_products = torch.sum(torch.mul(u_emb, v_emb), dim=1)
        return -((dot_products.sigmoid() + self.eps).log().sum())

    def sg_negative_loss(self, source, target):
        u_emb = self.embedding_user(source.long())
        v_emb = self.embedding_user(target.long())
        dot_products = torch.sum(torch.mul(u_emb, v_emb), dim=1)
        return -((1 - dot_products.sigmoid() + self.eps).log().sum())

    def dimension_reg(self):
        # when alpha = 0, p_vec should be a vector of all ones
        p_vec = self.degrees.pow(self.alpha)
        p_vec /= p_vec.sum()
        p_vec *= self.num_users

        col_sums = torch.matmul(self.embedding_user.weight.t(), p_vec)
        return self.lam * col_sums.norm(2).pow(2)

    def freeze_embeddings(self):
        for param in self.embedding_user.parameters():
            param.requires_grad = False

    def unfreeze_embeddings(self):
        for param in self.embedding_user.parameters():
            param.requires_grad = True

    def _get_edge_features(self, src, tgt, method):
        """
        Generate features for edges from the node embeddings.

        Args:
        - src (torch.Tensor): Source nodes
        - tgt (torch.Tensor): Target nodes
        - method (str): How to combine node embeddings ("concatenate", "hadamard", "average")
        
        Returns:
        - torch.Tensor: Edge features
        """
        src_emb = self.embedding_user(src.long())
        tgt_emb = self.embedding_user(tgt.long())

        if method == "concatenate":
            return torch.cat([src_emb, tgt_emb], dim=1)
        elif method == "hadamard":
            return src_emb * tgt_emb
        elif method == "average":
            return (src_emb + tgt_emb) / 2
        else:
            raise ValueError("Unsupported feature combination method")

    # TODO: give the classifier output
    def forward(self, src, tgt, method="hadamard"):
        edge_features = self._get_edge_features(src, tgt, method)
        return self.classifier(edge_features)
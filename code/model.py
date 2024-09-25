"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
# import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import world


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
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, 
            embedding_dim=self.latent_dim,
            device=self.device)        

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
        p_vec *= self.n_users

        col_sums = torch.matmul(self.embedding_user.weight.t(), p_vec)
        return self.lam * col_sums.norm(2).pow(2)

    def forward(self, src, tgt):
        src = src.long()
        tgt = tgt.long()
        src_emb = self.embedding_user(src)
        tgt_emb = self.embedding_user(tgt)
        scores = torch.sum(src_emb*tgt_emb, dim=1)
        return scores
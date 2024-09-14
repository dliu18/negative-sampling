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


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def sg_positive_loss(self, source, target):
        raise NotImplementedError

    def sg_negative_loss(self, source, target):
        raise NotImplementedError

    def dimension_reg(self):
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.latent_dim = config['latent_dim_rec']
        self.lam = config["reg_lam"]
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        

    def sg_positive_loss(self, source, target):
        u_emb = self.embedding_user(source.long())
        v_emb = self.embedding_user(target.long())
        dot_products = torch.sum(torch.mul(u_emb, v_emb), dim=1)
        return -1 * dot_products.sigmoid().sum()

    def sg_negative_loss(self, source, target):
        u_emb = self.embedding_user(source.long())
        v_emb = self.embedding_user(target.long())
        neg_dot_products = -torch.sum(torch.mul(u_emb, v_emb), dim=1)
        return -(neg_dot_products.sigmoid().sum())

    def dimension_reg(self):
        col_sums = torch.sum(self.embedding_user.weight, dim=0)
        return col_sums.norm(2).pow(2)

    def forward(self, src, dst):
        src = src.long()
        dst = dst.long()
        src_emb = self.embedding_user(src)
        dst_emb = self.embedding_item(dst)
        scores = torch.sum(users_emb*dst_emb, dim=1)
        return self.f(scores)
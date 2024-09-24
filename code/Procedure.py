'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from evaluator import Evaluator
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2


# expects the loss to be either SkipGramAugmentedLoss or SkipGramLoss
def train(dataset, sg_model, loss_obj, epoch, writer=None):
    sg_model.train() # puts the model in training mode
    loss_obj: utils.Loss = loss_obj

    loader = None
    if world.config["base_model"] == 'n2v':
        loader = dataset.get_train_loader_rw(
            batch_size = world.config['batch_size'], 
            sample_negatives = True)
    elif world.config["base_model"] == 'line':
        loader = dataset.get_train_loader_edges(
            batch_size = world.config['batch_size'], 
            sample_negatives = True)

    num_users = dataset.n_users
    total_batch = num_users // world.config['batch_size'] + 1
    aver_loss = 0.
    batch_i = 0

    for pos_sample, _ in tqdm(loader):
        # each row of pos and neg samples is of the form [src, dst1, dst2, ...]
        batch_pos = pos_sample[:, 1:].reshape(-1).to('cuda')
        batch_neg = dataset.get_sg_negatives(
            shape = (world.config["K"] * len(batch_pos),),
            alpha = world.config["alpha"]).to('cuda')
        batch_users = pos_sample[:, 0].reshape(-1).to('cuda')

        pos_loss, neg_loss, dimension_regularization = loss_obj.stageOne(epoch,
            batch_i, 
            batch_users, 
            batch_pos, 
            batch_neg)
        aver_loss += (pos_loss + neg_loss)
        if world.tensorboard:
            writer.add_scalar(f'Loss/positive_loss', pos_loss, epoch * int(num_users / world.config['batch_size']) + batch_i)
            writer.add_scalar(f'Loss/negative_loss', neg_loss, epoch * int(num_users / world.config['batch_size']) + batch_i)
            writer.add_scalar(f'Loss/total_loss', pos_loss + neg_loss, epoch * int(num_users / world.config['batch_size']) + batch_i)
            writer.add_scalar(f'Loss/dimension_regularization', dimension_regularization, epoch * int(num_users / world.config['batch_size']) + batch_i)
        batch_i += 1
    aver_loss = aver_loss / total_batch
    return f"loss: {aver_loss:,}"
    
     
def test(dataset, sg_model, epoch, writer):
    max_memory = torch.cuda.max_memory_allocated(device=torch.device("cuda"))
    if writer:
        writer.add_scalar("GPU usage/memory (bytes)", max_memory, epoch)

    for test_set in ["valid", "test"]:
        test_data = None
        if test_set != "test" and test_set != "valid":
            raise NotImplementedError("The provided test set is not valid")
        else:
            if test_set == "test":    
                test_data = dataset.get_test_data()
            elif test_set == "valid":
                test_data = dataset.get_valid_data()

        # test AUC
        label, avg_auc = Evaluator.test_auc(sg_model, dataset, test_set)
        if writer:
            writer.add_scalar(f'metrics/{test_set}/{label}', avg_auc, epoch)

        # test MRR
        label, all_mrr = Evaluator.test_mrr(sg_model, dataset, test_set)
        if writer:
            writer.add_scalar(f'metrics/{test_set}/{label}', all_mrr.mean(), epoch)

        # test Hits@k
        label, all_hits = Evaluator.test_hits(sg_model, dataset, test_set)
        if writer:
            writer.add_scalar(f'metrics/{test_set}/{label}', all_hits.mean(), epoch)

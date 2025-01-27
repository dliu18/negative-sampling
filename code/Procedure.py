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
def train(dataset, sg_model, loss_obj, epoch, completed_batches, writer=None):
    sg_model.train() # puts the model in training mode
    loss_obj: utils.Loss = loss_obj

    loader = None
    if world.config["base_model"] == 'n2v':
        loader = dataset.get_train_loader_rw(
            batch_size = world.config['batch_size'], 
            sample_negatives = True,
            p=world.config["n2v_p"],
            q=world.config["n2v_q"])
    elif world.config["base_model"] == 'line':
        loader = dataset.get_train_loader_edges(
            batch_size = world.config['batch_size'], 
            sample_negatives = True)

    num_edges = dataset.n_train_edges
    total_batch = num_edges // world.config['batch_size'] + 1
    aver_loss = 0.

    for pos_sample, _ in tqdm(loader):
        # each row of pos and neg samples is of the form [src, dst1, dst2, ...]
        batch_pos = pos_sample[:, 1:].reshape(-1).to('cuda')
        batch_neg = dataset.get_sg_negatives(
            shape = (world.config["K"] * len(batch_pos),),
            alpha = world.config["alpha"]).to('cuda')
        batch_users = pos_sample[:, 0].reshape(-1).to('cuda')

        pos_loss, neg_loss, dimension_regularization = loss_obj.stageOne(epoch,
            completed_batches, 
            batch_users, 
            batch_pos, 
            batch_neg)
        aver_loss += (pos_loss + neg_loss)
        if world.tensorboard:
            writer.add_scalar(f'Loss/positive_loss', pos_loss, completed_batches)
            writer.add_scalar(f'Loss/negative_loss', neg_loss, completed_batches)
            writer.add_scalar(f'Loss/total_loss', pos_loss + neg_loss, completed_batches)
            writer.add_scalar(f'Loss/dimension_regularization', dimension_regularization, completed_batches)
        completed_batches += 1
    aver_loss = aver_loss / total_batch
    return f"loss: {aver_loss:,}", completed_batches
    
def train_edge_classifier(dataset, sg_model, loss_obj, writer=None, epochs=5, plot=False):
    sg_model.train()
    sg_model.freeze_embeddings()
    loss_obj.reset_classifier_optimization()

    batch_size = world.config['batch_size']
    loader = dataset.get_train_loader_edges(
        batch_size = batch_size, 
        sample_negatives = True)

    num_edges = dataset.n_train_edges
    total_batch = num_edges // batch_size + 1
    for epoch in tqdm(range(epochs)):
        if epoch % 1 == 0:
            test(dataset, sg_model, epoch, writer, 
                prefix="classifier/", print_result=False, use_classifier=True)
        batch_i = 0
        for pos_sample, _ in loader:
            pos_source = pos_sample[:, 0].reshape(-1).to('cuda')
            pos_target = pos_sample[:, 1:].reshape(-1).to('cuda')
            
            # neg_source = dataset.get_sg_negatives(
            #     shape = (len(pos_source)*5,),
            #     alpha = 0.0).to('cuda')
            neg_source = pos_source.repeat_interleave(1)
            neg_target = dataset.get_sg_negatives(
                shape = (len(pos_source),),
                alpha = 0.0).to('cuda')

            classifier_loss = loss_obj.CrossEntropyLoss(pos_source, pos_target, neg_source, neg_target)
            if world.tensorboard and plot:
                writer.add_scalar(f'Loss/classifier_loss', classifier_loss, epoch * total_batch + batch_i)
            batch_i += 1

    sg_model.unfreeze_embeddings()


def test(dataset, sg_model, epoch, writer, prefix="", print_result=True, use_classifier=True):
    max_memory = torch.cuda.max_memory_allocated(device=torch.device("cuda"))
    if writer and prefix == "":
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
        label, avg_auc = Evaluator.test_auc(sg_model, dataset, test_set, use_classifier)
        if print_result:
            print(f'AUC: {avg_auc:.4f}')
        if writer:
            writer.add_scalar(prefix + f'metrics/{test_set}/{label}', avg_auc, epoch)

        # test MRR
        if dataset.dataset_name != "ogbl-ppa":
            label, all_mrr = Evaluator.test_mrr(sg_model, dataset, test_set, use_classifier)
            if print_result:
                print(f'MRR: {all_mrr.mean():.4f}')
            if writer:
                writer.add_scalar(prefix + f'metrics/{test_set}/{label}', all_mrr.mean(), epoch)

        # test Hits@k
        for K in [20, 50, 100]:
            label, all_hits = Evaluator.test_hits(sg_model, K, dataset, test_set, use_classifier)
            if print_result:
                print(f'Hits@{K}: {all_hits.mean():.4f}')
            if writer:
                writer.add_scalar(prefix + f'metrics/{test_set}/{label}', all_hits.mean(), epoch)

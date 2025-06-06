import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import BasicModel, SGModel
from world import config
from sklearn.metrics import roc_auc_score
import random
import os

class Loss:    
    def __init__(self,
                 sg_model : BasicModel,
                 config : dict):
        self.model = sg_model
        self.lr = config['lr']
        self.opt = optim.Adam(sg_model.get_embeddings().parameters(), lr=self.lr)
        self.opt_classifier = optim.Adam(sg_model.get_classifier().parameters(), lr=self.lr)

    def stageOne(self, epoch, batch_num, users, pos, neg):
        raise NotImplementedError

    def reset_classifier_optimization(self):
        self.opt_classifier = optim.Adam(self.model.get_classifier().parameters(), lr=self.lr * 0.1) #0.00001

    def CrossEntropyLoss(self, pos_users, pos_targets, neg_users, neg_targets):
        pos_loss = -torch.log(self.model(pos_users, pos_targets, use_classifier=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.model(neg_users, neg_targets, use_classifier=True) + 1e-15).mean()
        total_loss = pos_loss + neg_loss

        self.opt_classifier.zero_grad()
        total_loss.backward()
        self.opt_classifier.step()

        return pos_loss.detach().to('cpu') + neg_loss.detach().to('cpu')

class SkipGramLoss(Loss):
    def __init__(self,
                 sg_model : SGModel,
                 config: dict):
        super(SkipGramLoss, self).__init__(sg_model, config)

    def stageOne(self, epoch, batch_num, users, pos, neg):
        assert len(pos) % len(users) == 0
        assert len(neg) % len(users) == 0

        pos_users = users.repeat_interleave(int(len(pos) / len(users)))
        pos_loss = self.model.sg_positive_loss(pos_users, pos)

        neg_users = users.repeat_interleave(int(len(neg) / len(users)))
        neg_loss = self.model.sg_negative_loss(neg_users, neg) / world.config["K"]
        dimension_regularization = self.model.dimension_reg()
        total_loss = pos_loss + neg_loss

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        return pos_loss.detach().to('cpu'), neg_loss.detach().to('cpu'), dimension_regularization.detach().to('cpu')

class SkipGramAugmentedLoss(Loss):
    def __init__(self,
                 sg_model : SGModel,
                 config: dict):
        super(SkipGramAugmentedLoss, self).__init__(sg_model, config)

    def stageOne(self, epoch, batch_num, users, pos, neg):
        assert len(pos) % len(users) == 0

        pos_users = users.repeat_interleave(int(len(pos) / len(users)))
        pos_loss = self.model.sg_positive_loss(pos_users, pos)
        # neg_loss = self.model.sg_negative_loss(users, neg)
        dimension_regularization = self.model.dimension_reg()

        self.opt.zero_grad()
        pos_loss.backward()
        self.opt.step()

        assert "n_negative" in config
        # if epoch % config['n_negative'] == 0 and batch_num == 0:
        if batch_num % config['n_negative'] == 0 and batch_num > 0:
            self.opt.zero_grad()
            dimension_regularization.backward()
            self.opt.step()

        return pos_loss.detach().to('cpu'), \
                0.0, \
                dimension_regularization.detach().to('cpu') / config["lambda"]

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    file = f"{world.config['base_model']}-{world.config['loss_func']}-{world.config['n_negative']}-{world.config['lambda']}-{world.dataset}.pth.tar"
    return os.path.join(world.FILE_PATH,file)

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================

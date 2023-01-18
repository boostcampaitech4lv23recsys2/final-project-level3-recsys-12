import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import warnings
from data_loader.AE_dataloader import *

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=True)

class LossFunc(nn.Module):

    def __init__(self, loss_type = 'Multinomial', model_type = None):
        super(LossFunc, self).__init__()
        self.loss_type = loss_type
        self.model_type = model_type

    def forward(self, recon_x = None, x = None, mu = None, logvar = None, anneal = None):
        if self.loss_type == 'Gaussian':
            loss = self.Gaussian(recon_x, x)
        elif self.loss_type == 'Logistic':
            loss = self.Logistic(recon_x, x)
        elif self.loss_type == 'Multinomial':
            loss = self.Multinomial(recon_x, x)
        else:
            raise Exception("Not correct loss_type!")
        
        if self.model_type == 'VAE':
            KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = loss + anneal * KLD
        
        return loss

    def Gaussian(self, recon_x, x):
        gaussian = F.mse_loss(recon_x, x)
        return gaussian

    def Logistic(self, recon_x, x):
        logistic = F.binary_cross_entropy(recon_x.sigmoid(), x, reduction='none').sum(1).mean()
        return logistic

    def Multinomial(self, recon_x, x):
        multinomial = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        return multinomial


def get_ndcg(pred_list, true_list):
    idcg = sum((1 / np.log2(rank + 2) for rank in range(1, len(pred_list))))
    dcg = 0
    for rank, pred in enumerate(pred_list):
        if pred in true_list:
            dcg += 1 / np.log2(rank + 2)
    ndcg = dcg / idcg
    return ndcg

def get_hit(pred_list, true_list):
    hit_list = set(true_list) & set(pred_list)
    hit = len(hit_list) / len(true_list)
    return hit

def get_recall(pred_list, true_list, topk=10):
    pred = pred_list[:topk]
    num_hit = len(set(pred).intersection(set(true_list)))
    recall = float(num_hit) / len(true_list)
    return recall

def train(model, criterion, optimizer, data_loader, make_matrix_data_set, config):
    update_count = 1
    model.train()
    loss_val = 0
    best_loss = float("inf")
    for users in data_loader:
        mat = make_matrix_data_set.make_matrix(users)
        mat = mat.to(config.device)

        if config.model == 'MultiVAE':
            anneal = min(config.anneal_cap, 1. * update_count / config.total_anneal_steps)
            update_count += 1
            recon_mat, mu, logvar = model(mat, loss=True)
            optimizer.zero_grad()
            loss = criterion(recon_mat, mat, mu, logvar, anneal)

        else:
            recon_mat = model(mat)
            optimizer.zero_grad()
            loss = criterion(recon_x = recon_mat, x = mat)

        if best_loss > loss:
            torch.save(model.state_dict(), f"/opt/ml/input/model/saved_model/best_model.pt")
        
        loss_val += loss.item()

        loss.backward()
        optimizer.step()

    loss_val /= len(data_loader)

    return loss_val

def evaluate(model, data_loader, user_train, user_valid, make_matrix_data_set, config):
    model.eval()

    NDCG = 0.0 # NDCG@10
    HIT = 0.0 # HIT@10
    RECALL = 0.0 # RECALL@10
    with torch.no_grad():
        for users in data_loader:
            mat = make_matrix_data_set.make_matrix(users)
            mat = mat.to(config.device)

            recon_mat = model(mat)
            recon_mat[mat == 1] = -np.inf
            rec_list = recon_mat.argsort(dim = 1)

            for user, rec in zip(users, rec_list):
                uv = user_valid[user.item()]
                up = rec[-10:].cpu().numpy().tolist()
                NDCG += get_ndcg(pred_list = up, true_list = uv)
                HIT += get_hit(pred_list = up, true_list = uv)
                RECALL += get_recall(pred_list = up, true_list = uv)

    NDCG /= len(data_loader.dataset)
    HIT /= len(data_loader.dataset)
    RECALL /= len(data_loader.dataset)

    return NDCG, HIT, RECALL

def logging_time(original_fn):
    import time
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn

@ logging_time
def make_submission(model, data_loader, user_decoder, item_decoder, make_matrix_data_set, config):
    model.eval()

    with torch.no_grad():
        prediction = []
        for users in data_loader:
            mat = make_matrix_data_set.make_matrix(users, train=False)
            mat = mat.to(config.device)
            recon_mat = model(mat)
            recon_mat[mat == 1] = -np.inf
            rec_list = recon_mat.argsort(dim=1)
            
            for user, rec in zip(users, rec_list):
                up = rec[-config.topk:].cpu().numpy().tolist()
                for item in up[::-1]:
                    prediction.append([user_decoder[int(user[0])], item_decoder[item]])
    return prediction

@ logging_time
def inference(config, model, user_selected_data, item_decoder):
    model.eval()
    with torch.no_grad():
        prediction = []
        model_output = model(user_selected_data.type(torch.FloatTensor).to(config.device))
        model_output[user_selected_data == 1] = -np.inf
        recommend_res = model_output.argsort(dim=1).squeeze()
        up = recommend_res[-config.topk:].cpu().numpy().tolist()
        for item in up[::-1]:
            prediction.append(item_decoder[item])
        return prediction
    
def make_single_dummy_user_input(df, item_encoder, n_interaction = 50):
    dummy_input = torch.zeros((df.item.nunique()), dtype=torch.int64)
    
    dummy_indices = torch.tensor(df.item.unique())
    sample_indices = torch.randint(0, len(dummy_indices), (1, n_interaction)).squeeze()
    dummy_indices = dummy_indices[sample_indices].apply_(lambda x: item_encoder[x]).unsqueeze(dim=0)
    
    dummy_input[dummy_indices] = 1
    return dummy_input.unsqueeze(dim=0)

def generate_encoder_decoder(df, col: str) -> dict:
    """
    encoder, decoder 생성

    Args:
        col (str): 생성할 columns 명
    Returns:
        dict: 생성된 user encoder, decoder
    """

    encoder = {}
    decoder = {}
    ids = df[col].unique()

    for idx, _id in enumerate(ids):
        encoder[_id] = idx
        decoder[idx] = _id

    return encoder, decoder


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
import os 
import itertools
import pickle
from dotenv import load_dotenv

from tqdm import tqdm
import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

class torchDataset(Dataset):
    def __init__(self, total, user_embeddings, item_embeddings):
        self.total = np.array(total)
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

    def __getitem__(self, idx:int):
        i, v, y = int(self.total[idx, 0]), int(self.total[idx, 1]), int(self.total[idx, -1])
        user, item = self.user_embeddings[i], self.item_embeddings[v]
        user, item = torch.tensor(user), torch.tensor(item)
        """ user, item returns list of tensors """
        return i, v, user, item, y

    def __len__(self):
        return self.total.shape[0]

class DeepFMDataset(pl.LightningDataModule):
    def __init__(self, train, val, user_embeddings, item_embeddings, uid, iid, 
            user_col, item_col, batch_size=512, test=None):
        super().__init__()
        self.uid = uid
        self.iid = iid

        self.user_col = user_col
        self.item_col = item_col

        self.train = self._get_total(train)
        self.val = self._get_total(val)
        self.test = test

        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

        self.setup()

        self.batch_size = batch_size

    def setup(self, stage=None):
        with open(self.user_embeddings, 'rb') as f:
            self.user_embeddings = pickle.load(f)

        with open(self.item_embeddings, 'rb') as f:
            self.item_embeddings = pickle.load(f)
    
    def _get_total(self, data):
        try:
            data[self.user_col] = [self.uid[x] for x in data[self.user_col]]
            data[self.item_col] = [self.iid[x] for x in data[self.item_col]]
        except:
            pass
        return data

    def train_dataloader(self):
        return DataLoader(torchDataset(self.train, self.user_embeddings, self.item_embeddings), 
                batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(torchDataset(self.val, self.user_embeddings, self.item_embeddings), 
                batch_size=self.batch_size)
    
    def test_dataloader(self, dataset=False):
        assert type(self.test) == pd.DataFrame, "You Didn't Pass Test Dataset or Pass in Wrong Format, Expect pd.DataFrame object"
        return DataLoader(torchDataset(self.test, self.user_embeddings, self.item_embeddings), 
                batch_size=self.batch_size)

class DeepFactorizationMachine(pl.LightningModule):
    def __init__(self, env='1.2', hidden_layer_dims=[], input_dim=0, max_score=3):
        super(DeepFactorizationMachine, self).__init__()

        self.input_dim = input_dim
        self.hidden_layer_dims = [self.input_dim if i == 0 else hidden_layer_dims[i-1] for i in range(len(hidden_layer_dims)+1)]
        self.max_score = max_score

        params = []
        for i in range(len(self.hidden_layer_dims)-1):
            params.append(nn.Linear(self.hidden_layer_dims[i], self.hidden_layer_dims[i+1]))
            params.append(nn.BatchNorm1d(self.hidden_layer_dims[i+1]))
            params.append(nn.Dropout(0.2))
            params.append(nn.ReLU())

        self.MLP = nn.Sequential(*params)

        self.loss_func = F.binary_cross_entropy_with_logits


    def _init_weights(self, model):
        if type(model) == nn.Linear:
            nn.init.xavier_normal(model.weight)
            model.bias.data.fill_(0.01)

    def _vec_to_device(self, vec):
        vec = vec.to(device=self.device, dtype=torch.float)
        return vec

    def _split_out(self, user, item):
        user_vec, item_vec = [], []
        for i in range(int(user.shape[1]/16)):
            user_vec.append(user[:, 16*i:16*(i+1)])
            item_vec.append(item[:, 16*i:16*(i+1)])

        v = 0
        for i in user_vec:
            for j in item_vec:
                out = i * j 
                if v == 0:
                    output = out
                else:
                    output = torch.cat((output, out), axis=1)
                
                v += 1
        return output

    def forward(self, user, item, y=None):
        user, item = self._vec_to_device(user), self._vec_to_device(item)
        if y != None:
            y = self._vec_to_device(y)

        mlp_input = torch.cat((user, item), axis=1)
        mlp_input = self.MLP(mlp_input)

        fm = self._split_out(user, item)
        fm = nn.Sequential(
            nn.Linear(fm.shape[1], fm.shape[1]),
            nn.BatchNorm1d(fm.shape[1]),
            nn.ReLU()
        )(fm)

        general_input = torch.cat((fm, mlp_input), axis=1)
        output = nn.Sequential(
            nn.Linear(general_input.shape[1], 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )(general_input).squeeze(-1)

        output = torch.where(torch.isnan(output), torch.full_like(output, 1e-6), output)
        output = torch.clamp(output, 1e-6, 1)

        if y != None:
            loss = self.loss_func(output, y)
            #output = self.classify(output)
            return output, loss
        else:
            return output
    
    def training_step(self, batch, batch_idx):
        i, v, user, item, y = batch 

        total, loss = self.forward(user, item, y)
        self.log('training_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        i, v, user, item, y = batch 

        total, loss = self.forward(user, item, y)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), weight_decay=0.0, lr=1e-2)
        return optimizer
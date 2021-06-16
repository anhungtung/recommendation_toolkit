import os
import random
import shutil
import itertools
import json
import pickle
import warnings

import numpy as np 
import pandas as pd
from dotenv import load_dotenv

from sklearn.utils import shuffle

from tqdm import tqdm
import torch
import torch.nn as nn

import matplotlib.pyplot as plt 
import seaborn as sns

import pytorch_lightning as pl 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from .recommendations_model.NeuCF import NeuCFDataset, NeuralMatrixFactorization
from .recommendations_model.DeepFM import DeepFMDataset, DeepFactorizationMachine

class BuildModel():
    def __init__(self, train, val, uid, iid, test=None, model_type='NeuCF', if_production=False,
                model_params='params/model_params.json', env_params='params/RECOMMEND.env'):
        '''
        Train & Make Predictions
        '''

        load_dotenv(env_params)

        if type(uid) == str:
            with open(uid, 'rb') as f:
                self.uid = pickle.load(f)
        else:
            self.uid = uid

        if type(iid) == str:
            with open(iid, 'rb') as f:
                self.iid = pickle.load(f)
        else:
            self.iid = iid

        self.user_col = os.getenv("USER_COLUMN")
        self.item_col = os.getenv("ITEM_COLUMN")
        self.rating_col = os.getenv("RATING_COLUMN")
        self.save_path = os.getenv("MODEL_SAVEPATH")
        
        self.train = self.transform(train) 
        self.val = self.transform(val) 
        self.test = self.transform(test)

        self.model_type = model_type

        if if_production:
            self.train = pd.concat([self.train, self.val])

        self.user_embeddings = os.getenv("USER_EMBEDDINGS")
        self.item_embeddings = os.getenv("ITEM_EMBEDDINGS")

        with open(model_params, 'rb') as f:
            self.params = json.load(f)
        
        self.max_score = self.params['max_score']
        
        self.batch_size = self.params['batch_size'] if 'batch_size' in self.params.keys() else 512
        self.input_dim = self.params['input_dim'] if 'input_dim' in self.params.keys() else 128
        self.hidden_layer_dims = eval(self.params['hidden_layer_dims']) if 'hidden_layer_dims' in self.params.keys() else [self.input_dim, self.input_dim // 2]
        self.max_epochs = self.params['max_epochs'] if 'max_epochs' in self.params.keys() else 10
        self.patience = self.params['patience'] if 'patience' in self.params.keys() else self.max_epochs // 3
        self.min_delta = self.params['min_delta'] if 'min_delta' in self.params.keys() else 1e-5
        self.class_weight = eval(self.params['class_weight']) if 'class_weight' in self.params.keys() else [1] * len(self.hidden_layer_dims)
        self.dropout = self.params['dropout'] if 'dropout' in self.params.keys() else 0.3
        self.output_thres = self.params['output_thres'] if 'output_thres' in self.params.keys() else 0.5
        self.lr_rate = self.params['lr_rate'] if 'lr_rate' in self.params.keys() else 0.5

        self.n_users = len(self.uid)
        self.n_items = len(self.iid)

        self._create_dir()
        self._clean_lightning_log()

    def _create_dir(self):
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
    
    def _clean_lightning_log(self):
        if os.path.isdir('lightning_logs/'):
            shutil.rmtree('lightning_logs/')
        
        if not os.path.isdir('lightning_logs/'):
            os.mkdir('lightning_logs/')
            
        return 
    
    def _getembeddings(self, path):
        with open(path, 'rb') as f:
            path = pickle.load(f)
        return path
    
    def _init_dataset_and_model(self):
        if self.model_type == 'NeuCF':    
            self.dataset = NeuCFDataset(self.train, self.val, self.user_embeddings, self.item_embeddings, 
                        self.uid, self.iid, user_col=self.user_col,
                        item_col=self.item_col, batch_size=self.batch_size, test=self.test
            )

            self.model = NeuralMatrixFactorization(
                hidden_layer_dims=self.hidden_layer_dims, input_dim=self.input_dim, max_score=self.max_score,
                dropout=self.dropout
            )
        
        elif self.model_type == 'DeepFM':
            self.dataset = DeepFMDataset(self.train, self.val, self.user_embeddings, self.item_embeddings, 
                            self.uid, self.iid, user_col=self.user_col,
                            item_col=self.item_col, batch_size=self.batch_size, test=self.test
                        )
        
            self.model = DeepFactorizationMachine(
                hidden_layer_dims=self.hidden_layer_dims, input_dim=self.input_dim, max_score=self.max_score
            )
        
        else:
            raise ValueError('Unknown arguments for model_type')

        
        self.early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=False,
            mode='min'
        )

        self.checkpoint_callback = ModelCheckpoint(
            filepath=self.save_path,
            save_top_k=0,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=''
        )
    
    def fit(self, save=True):
        self._init_dataset_and_model()
        self.trainer = pl.Trainer(max_epochs=self.max_epochs, callbacks=[self.early_stop_callback, self.checkpoint_callback])
        self.trainer.fit(self.model, self.dataset)
        
        if save:
            self.trainer.save_checkpoint(self.save_path + self.model_type + '.ckpt')
            torch.save(self.model, self.save_path + self.model_type + '.pkl')

    def predict_all(self, top_n=30):
        self._init_dataset_and_model()
        
        self.model = torch.load(self.save_path + self.model_type + '.pkl')
        self.model.eval()
        torch.no_grad()

        dataset = self.dataset.test_dataloader()

        user_indexing = self.test.groupby(by=self.user_col)[self.item_col].count().reset_index()
        inv_uid = {v : k for k, v in self.uid.items()}
        inv_iid = {v : k for k, v in self.iid.items()}
        
        tmp_dir, Rating, item_list, self.results = self.save_path + 'tmp.pickle', [], [], {}
        now_user = 0
        for idx, batch in enumerate(tqdm(dataset, desc='Prediction (Model)!!', disable=os.environ.get("DISABLE_TQDM", False))):
            i, v, user, item, y = batch
            output = self.model(user, item)

            Rating = [*Rating, *list(output.cpu().detach().numpy())]
            item_list = [*item_list, *list(v)]
            del output
            
            to_add = len(Rating) // self.batch_size
            
            for v in range(to_add):
                index = user_indexing.iloc[now_user, 1]

                scores, items = Rating[:index], item_list[:index]

                df = pd.DataFrame(items, columns=[self.item_col])
                df[self.rating_col] = scores

                df = df.sort_values(by=self.rating_col, ascending=False)
                df = df.iloc[:min(top_n, df.shape[0]), :]
                df = df[df[self.rating_col] > self.output_thres]
                recom = [inv_iid[x] for x in list(df[self.item_col])]

                self.results[now_user] = {
                    'Recommendations' : recom,
                    'Parameters' : {
                        'Average Score' : df[self.rating_col].mean(),
                        'Num' : df.shape[0],
                        'Model Name' : self.model_type
                    }
                }
                
                now_user += 1
                Rating = Rating[index:]
                item_list = item_list[index:]

        df = pd.DataFrame(item_list, columns=[self.item_col])
        df[self.rating_col] = Rating

        df = df.sort_values(by=self.rating_col, ascending=False)
        df = df.iloc[:min(top_n, df.shape[0]), :]
        df = df[df[self.rating_col] > self.output_thres]
        recom = [inv_iid[x] for x in list(df[self.item_col])]

        self.results[now_user] = {
            'Recommendations' : recom,
            'Parameters' : {
                'Average Score' : df[self.rating_col].mean(),
                'Num' : df.shape[0],
                'Model Name' : self.model_type
            }
        }

        with open(self.save_path + 'Prediction_Results.json', 'w') as outfile:
            json.dump(self.results, outfile)

        return
    
    def transform(self, data):
        try:
            data[self.user_col] = [self.uid[x] for x in data[self.user_col]]
            data[self.item_col] = [self.iid[x] for x in data[self.item_col]]
        except:
            pass
        return data
    
    def validate_user(self, scaledown=True):
        self.model = torch.load(self.save_path + self.model_type + '.pkl')
        self.model.eval()

        user_embed = self._getembeddings(self.user_embeddings)
        item_embed = self._getembeddings(self.item_embeddings)

        tester = shuffle(self.val.copy())

        res = []
        for i in tqdm(range(tester.shape[0]), desc='Validating ... ', disable=os.environ.get("DISABLE_TQDM", False)):

            user, item, y = tester.iloc[i, 0], tester.iloc[i, 1], tester.iloc[i, -1]
            
            user_emb, item_emb = user_embed[user], item_embed[item]
            user_emb = torch.tensor(user_emb).unsqueeze(0)
            item_emb = torch.tensor(item_emb).unsqueeze(0)

            output = self.model(user_emb, item_emb)
            result = float(output[0].detach().numpy())

            res.append((result, y))

        res = pd.DataFrame(res, columns=['Predicted', 'Real'])

        for item in set(res['Real']):
            sns.displot(data=res[res['Real']==item], x='Predicted', kind='kde')
            plt.title('Category : ' + str(item))
            plt.show()

        sns.displot(data=res, x='Predicted', hue='Real')
        plt.show()
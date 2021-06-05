import os 
import json
import pickle
import itertools
from dotenv import load_dotenv

import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch 
import torch.nn as nn 

import networkx as nx
from models.embeddings.embedding_models import Node2Vec, Word2VecEmbeddings

import warnings
warnings.filterwarnings("ignore")

def _create_dirs():
    if not os.path.isdir('Raw/Model_Data'):
        os.mkdir('Raw/Model_Data')

class ItemEmbeddings(nn.Module):
    """
        Node2Vec embeds item graph data 

        item_data : behavior_data for your items
        dim_data : categorical_data for your items
        
    """
    def __init__(self, item_data, dim_data, settings_params='params/ItemEmbeddings.env', 
            embeddings_params='params/Itemembeddings_params.json'):
        super(ItemEmbeddings, self).__init__()

        load_dotenv(settings_params, override=True)
        
        self.data = data 
        self.dim = dim_data
        self.user_col = os.getenv("USER_COLUMN")
        self.item_col = os.getenv("ITEM_COLUMN")
        self.rating_col = os.getenv("RATING_COLUMN")
        self.output_dir = os.getenv("OUTPUT_DIR")

        self.uid = {v : k for k, v in enumerate(set(self.data[self.user_col]))}
        self.iid = {v : k for k, v in enumerate(set(self.data[self.item_col]))}
        self.n_item = len(self.iid)

        with open(embeddings_params, 'rb') as f:
            self.params = json.load(f)

        self.levels = self.params['max_score'] if 'walk_length' in self.params.keys() else 20

        self.walk_length = self.params['walk_length'] if 'walk_length' in self.params.keys() else 20
        self.p = self.params['p_val'] if 'p_val' in self.params.keys() else 0.2
        self.q = self.params['q_val'] if 'q_val' in self.params.keys() else 0.2
        self.num_workers = self.params['num_workers'] if 'num_workers' in self.params.keys() else 1
        self.win_size = self.params['win_size'] if 'win_size' in self.params.keys() else 3
        self.n_iters = self.params['n_iters'] if 'n_iters' in self.params.keys() else 50
        self.embed_size = self.params['embed_size'] if 'embed_size' in self.params.keys() else 128

        try:
            self.levels = float(os.getenv("LEVELS"))
        except:
            self.levels = max(self.data[self.rating_col])
        
        self._create_edge_matrix()
        self.hparam = {}
        self.hparam['Order'] = []

    def _supports(self, data):
        data = np.dot(data.T, data)
        data = np.divide(data, self.n_item)
        return data 
    
    def _confidence(self, data):
        data = self._supports(data)
        conf = np.zeros(shape=(self.n_item, self.n_item))

        for i in range(self.n_item):
            for j in range(i, self.n_item):
                conf[i, j], conf[j, i] = 0, 0
                if j != i:
                    if data[j, j] > 0:
                        conf[i, j] = data[i, j]/data[j, j]
                    
                    if data[i, i] > 0:
                        conf[j, i] = data[i, j]/data[i, i]
        return conf
    
    def _create_edge_matrix(self):
        data = self.data.copy()

        data[self.item_col] = [self.iid[x] for x in data[self.item_col]]
        data[self.user_col] = [self.uid[x] for x in data[self.user_col]]
        data = data.drop_duplicates()

        data = pd.pivot_table(
            data=data, columns=self.item_col, index=self.user_col, values=self.rating_col,
            aggfunc='max',  fill_value=-1
        )

        self.edges = data

    def _create_edge_list(self, l):
        """ 
            Slice data with level of parameter l and create networkx edge_list
        """
        edge = np.where(self.edges == l, 1, 0)
        edge = self._confidence(edge)

        edge = pd.DataFrame(edge, index=range(self.n_item), columns=range(self.n_item))
        edge.columns = [x for x in range(self.n_item)]
        edge.index = [x for x in range(self.n_item)]
        
        G = nx.from_numpy_matrix(edge.values,
                                create_using=nx.DiGraph())
        return G
        
    def trainings(self, l, embeddings=None):
        G = self._create_edge_list(l=l)

        model = Node2Vec(
            graph=G, walk_length=self.walk_length, num_walks=self.num_walks, p=self.p,
            q=self.q, workers=self.num_workers
        )

        model.train(
            window_size=self.win_size, iter=self.n_iters, min_count=1, embed_size=self.embed_size
        )
        
        embeddings = model.get_embeddings(pre_trained=embeddings)

        return embeddings
    
    def positional_embeddings(self):
        self.pos = {}
        total_embedding_dim = int(self.embed_size/2)
        pos_dim, side_dim = int(total_embedding_dim/1.5), total_embedding_dim - int(total_embedding_dim/1.5)

        emb_layer = nn.Embedding(num_embeddings=self.n_item, embedding_dim=pos_dim, sparse=True)
        emb_layer_side = nn.Embedding(num_embeddings=self.dim.shape[1]-1, embedding_dim=side_dim)
        
        for item in range(self.n_item):
            one_hot = emb_layer(torch.tensor(item))
            side_index = list(self.dim[self.dim[self.item_col] == item].iloc[0, :-1]).index(1)
            dim_info = emb_layer_side(torch.tensor(side_index))

            self.pos[item] = torch.cat([one_hot, dim_info]).detach().numpy()

        self.hparam['Positional_Embedding'] = pos_dim
        self.hparam['Order'].append('Positional_Embedding')
        self.hparam['Side_Info_Embedding'] = side_dim
        self.hparam['Order'].append('Side_Info_Embedding')

    def _concat_embeddings(self):
        for item in tqdm(range(self.n_item), desc='Join Pooling Layer with Positional', disable=os.environ.get("DISABLE_TQDM", False)):
            if item in self.pos:
                self.all[item] = np.concatenate((self.all[item], self.pos[item]))
            else:
                self.all[item] = np.concatenate((self.all[item], [0] * int(self.embed_size/2)))

    def fit(self, dump=True):
        self.positional_embeddings()

        self.all = {}
        for l in tqdm(range(self.levels+1), desc='Create Item Embeddings...', disable=os.environ.get("DISABLE_TQDM", False)):
            if self.all == {}:
                embeddings = self.trainings(l=l, embeddings=None)
                self.all = embeddings.copy()
            else:
                self.all = self.trainings(l=l, embeddings=self.all)
            
            self.hparam['Social_Relation_Embedding_' + str(l)] = self.embed_size
            self.hparam['Order'].append('Social_Relation_Embedding_' + str(l))

        self._concat_embeddings()

        if dump:
            with open(self.output_dir, "wb") as outfile:
                pickle.dump(self.all, outfile)
            
            with open(self.output_dir[:-4] + '_iids.bin', "wb") as outfile:
                pickle.dump(self.iid, outfile)

        self.hparam['All_Dim'] = self.all[0].shape

        with open(self.output_dir + 'Item_Embedding_hparam.json', 'w') as outfile:
            json.dump(self.hparam, outfile)

        return self.all

class UserEmbeddings(nn.Module):
    """
        GRU4REC without session embeds user behavior sequence
    """
    def __init__(self, data, settings_params='params/UserEmbeddings.env', 
            embeddings_params='params/Userembeddings_params.json'):
        super(UserEmbeddings, self).__init__()

        load_dotenv(settings_params, override=True)
        
        self.data = data 

        self.user_col = os.getenv("USER_COLUMN")
        self.item_col = os.getenv("ITEM_COLUMN")
        self.rating_col = os.getenv("RATING_COLUMN")
        self.day_col = os.getenv("DATE_COLUMN")

        self.data = self.data.drop_duplicates(subset=[self.user_col, self.item_col, self.day_col])

        self.uid = {v : k for k, v in enumerate(set(self.data[self.user_col]))}
        self.iid = {v : k for k, v in enumerate(set(self.data[self.item_col]))}
        self.n_user = len(self.uid)
        self.n_item = len(self.iid)
        self.levels = max(self.data[self.rating_col])

        with open(embeddings_params, 'rb') as f:
            self.params = json.load(f)

        self.win_size = self.params['win_size'] if 'win_size' in self.params.keys() else 3
        self.n_iters = self.params['n_iters'] if 'n_iters' in self.params.keys() else 50
        self.embed_size = self.params['embed_size'] if 'embed_size' in self.params.keys() else 128

        _create_dirs()

        self.data[self.user_col] = [self.uid[x] for x in self.data[self.user_col]]
        self.data[self.item_col] = [self.iid[x] for x in self.data[self.item_col]]

        self.output_dir = os.getenv("OUTPUT_DIR")
        self.padding_length = int(os.getenv("PADDING_LENGTH"))

        self.hparam = {}
        self.hparam['Order'] = []

    def _create_user_sequences(self, l=-1, padding=False, all_item=False):
        if not all_item:
            data = self.data[self.data[self.rating_col] == l]
        else:
            data = self.data.copy()

        seq = {}
        for user, df in data.groupby(by=self.user_col):
            if all_item:
                seq[user] = [int(x) for x in list(df[self.item_col]*df[self.rating_col])]
            else:
                seq[user] = [str(x) for x in list(df[self.item_col])]

            if padding : 
                if len(seq[user]) < self.padding_length:
                    s = [-1] * (self.padding_length - len(seq[user]))
                    seq[user] = s + seq[user]
                else:
                    seq[user] = seq[user][-self.padding_length:]
        
        return seq 

    def _trainings(self, l, embeddings=None):
        seq = self._create_user_sequences(l, padding=False)
        
        model = Word2VecEmbeddings(
            seq=seq, n_item=self.n_item
        )

        model.train(
            window_size=self.win_size, iter=self.n_iters, min_count=1, embed_size=self.embed_size
        )
        
        embeddings = model.get_embeddings(pre_trained=embeddings)
        return embeddings

    def w2v_embeddings(self):
        self.all = {}
        for l in tqdm(range(self.levels+1), desc='Create User Embeddings...', disable=os.environ.get("DISABLE_TQDM", False)):
            if self.all == {}:
                embeddings = self._trainings(l=l, embeddings=None)
                self.all = embeddings.copy()
            else:
                self.all = self._trainings(l=l, embeddings=self.all)

    def positional_embeddings(self):
        seq = self._create_user_sequences(padding=True, all_item=True)
      
        emb_layer = nn.LSTM(self.padding_length, self.embed_size)
        hidden = (torch.randn(1, 1, self.embed_size).to(dtype=torch.float),
                torch.randn(1, 1, self.embed_size).to(dtype=torch.float))

        self.pos = {}
        
        for user in range(self.n_user):
            in_data = torch.tensor(seq[user], dtype=torch.float).view(1, 1, -1)
            out, hidden = emb_layer(in_data, hidden)
            self.pos[user] = out.detach().numpy()[0][0]
        
        self.hparam['Order'].append('Sequence_Embeddings')
        self.hparam['Sequence_Embeddings'] = self.embed_size
            
    def _poolings(self, l):
        data = self.data[self.data[self.rating_col] == l]
        
        part_pre = int(self.embed_size * l)
        part_last = int(self.embed_size * (l+1))

        for user in range(self.n_user):
            df = data[data[self.user_col] == user]

            if len(set(df[self.item_col])) > 0:
                pool = np.zeros(shape=(len(set(df[self.item_col])), self.embed_size))
            
                for i, item in enumerate(set(df[self.item_col])):
                    pool[i, :] = self.all[item][part_pre:part_last]
        
                avg_pool = np.average(pool, axis=0)
                mean_pool = np.mean(pool, axis=0)
            
            else:
                avg_pool, mean_pool = np.zeros(shape=(self.embed_size)), np.zeros(shape=(self.embed_size))

            if user in self.embeddings:
                self.embeddings[user] = np.concatenate((self.embeddings[user], np.concatenate((avg_pool, mean_pool))))
            else:
                self.embeddings[user] = np.concatenate((avg_pool, mean_pool))
    
        return self.embeddings

    def _concat_embeddings(self):
        for user in tqdm(range(self.n_user), desc='Join Pooling Layer with Positional', disable=os.environ.get("DISABLE_TQDM", False)):
            if user in self.pos:
                self.embeddings[user] = np.concatenate((self.embeddings[user], self.pos[user]))
            else:
                self.embeddings[user] = np.concatenate((self.embeddings[user], [0] * self.embed_size))
            
    def fit(self, dump=True):
        self.positional_embeddings()
        self.w2v_embeddings()
        self.embeddings = {}

        for l in tqdm(range(self.levels+1), desc='Pooling User Embeddings ...', disable=os.environ.get("DISABLE_TQDM", False)):
            self.embeddings = self._poolings(l)

            self.hparam['Order'].append('Item_Poolings_' + str(l))
            self.hparam['Item_Poolings_' + str(l)] = self.embed_size * 2
        
        self._concat_embeddings()

        if dump:
            with open(self.output_dir, "wb") as outfile:
                pickle.dump(self.embeddings, outfile)
            
            with open(self.output_dir[:-4] + '_uids.bin', "wb") as outfile:
                pickle.dump(self.uid, outfile)

        self.hparam['All_Dim'] = self.embeddings[0].shape

        with open(self.output_dir + 'User_Embedding_hparam.json', 'w') as outfile:
            json.dump(self.hparam, outfile)

        return self.embeddings
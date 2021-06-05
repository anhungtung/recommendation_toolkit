import itertools
import math
import random

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm import trange

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl 

from .alias import alias_sample, create_alias_table, partition_num

from gensim.models import Word2Vec

class RandomWalker:
    def __init__(self, G, p=1, q=1, use_rejection_sampling=0):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        :param use_rejection_sampling: Whether to use the rejection sampling strategy in node2vec.
        """
        self.G = G
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling

    def deepwalk_walk(self, walk_length, start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def node2vec_walk(self, walk_length, start_node):

        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],
                                                      alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def node2vec_walk2(self, walk_length, start_node):
        """
        Reference:
        KnightKing: A Fast Distributed Graph Random Walk Engine
        """

        def rejection_sample(inv_p, inv_q, nbrs_num):
            upper_bound = max(1.0, max(inv_p, inv_q))
            lower_bound = min(1.0, min(inv_p, inv_q))
            shatter = 0
            second_upper_bound = max(1.0, inv_q)
            if (inv_p > second_upper_bound):
                shatter = second_upper_bound / nbrs_num
                upper_bound = second_upper_bound + shatter
            return upper_bound, lower_bound, shatter

        G = self.G
        alias_nodes = self.alias_nodes
        inv_p = 1.0 / self.p
        inv_q = 1.0 / self.q
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    upper_bound, lower_bound, shatter = rejection_sample(
                        inv_p, inv_q, len(cur_nbrs))
                    prev = walk[-2]
                    prev_nbrs = set(G.neighbors(prev))
                    while True:
                        prob = random.random() * upper_bound
                        if (prob + shatter >= upper_bound):
                            next_node = prev
                            break
                        next_node = cur_nbrs[alias_sample(
                            alias_nodes[cur][0], alias_nodes[cur][1])]
                        if (prob < lower_bound):
                            break
                        if (prob < inv_p and next_node == prev):
                            break
                        _prob = 1.0 if next_node in prev_nbrs else inv_q
                        if (prob < _prob):
                            break
                    walk.append(next_node)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))
        walks = list(map(str, walks))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(
                        walk_length=walk_length, start_node=v))
                elif self.use_rejection_sampling:
                    walks.append(self.node2vec_walk2(
                        walk_length=walk_length, start_node=v))
                else:
                    walks.append(self.node2vec_walk(
                        walk_length=walk_length, start_node=v))
        return walks

    def get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        if not self.use_rejection_sampling:
            alias_edges = {}

            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                if not G.is_directed():
                    alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
                self.alias_edges = alias_edges

        self.alias_nodes = alias_nodes
        return

class Word2VecEmbeddings:
    def __init__(self, seq, n_item):
        self.seq = seq 
        self.n_item = n_item

        self.seq = [self.seq[item] for item in self.seq]
    
    def train(self, embed_size=128, window_size=5, workers=3, iterations=5, **kwargs):
        kwargs["sentences"] = self.seq
        kwargs["min_count"] = kwargs.get("min_count", 1)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iterations

        self.embed_size = embed_size

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        return model
    
    def get_embeddings(self, pre_trained=None):
        if self.w2v_model is None:
            print("model not train")
            return {}

        if type(pre_trained) != dict:
            self._embeddings = {}
        else:
            self._embeddings = pre_trained

        for word in range(self.n_item):
            try:
                if type(pre_trained) != dict:
                    self._embeddings[word] = self.w2v_model.wv[str(word)]
                else:
                    self._embeddings[word] = np.concatenate((self._embeddings[word], self.w2v_model.wv[str(word)]))
            except:
                if type(pre_trained) != dict:
                    self._embeddings[word] = [0] * self.embed_size
                else:
                    self._embeddings[word] = np.concatenate((self._embeddings[word], [0] * self.embed_size))

        return self._embeddings 

class Node2Vec:
    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, 
            workers=1, use_rejection_sampling=0):

        self.graph = graph
        self._embeddings = {}
        self.walker = RandomWalker(
            graph, p=p, q=q, use_rejection_sampling=use_rejection_sampling)

        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1
            )
        
        self.sentences = [eval(x) for x in self.sentences]
        self.sentences = [[str(x) for x in y] for y in self.sentences]
        
    def train(self, embed_size=128, window_size=5, workers=3, iterations=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 1)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iterations

        self.embed_size = embed_size

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        return model

    def get_embeddings(self, pre_trained=None):
        if self.w2v_model is None:
            print("model not train")
            return {}

        if type(pre_trained) != dict:
            self._embeddings = {}
        else:
            self._embeddings = pre_trained

        for word in self.graph.nodes():
            try:
                if type(pre_trained) != dict:
                    self._embeddings[word] = self.w2v_model.wv[str(word)]
                else:
                    self._embeddings[word] = np.concatenate((self._embeddings[word], self.w2v_model.wv[str(word)]))
            except:
                if type(pre_trained) != dict:
                    self._embeddings[word] = [0] * self.embed_size
                else:
                    self._embeddings[word] = np.concatenate((self._embeddings[word], [0] * self.embed_size))

        return self._embeddings 

class BayesianPersonalizedRankingLoss(nn.Module):
    """ Not Completed """
    def __init__(self):
        super(BayesianPersonalizedRankingLoss, self).__init__()

    def forward(self, output):
        diff = output.diag().view(-1, 1).expand_as(output) - output
        loss = -torch.mean(F.logsigmoid(diff))
        return loss

class __baseGRU4REC(pl.LightningModule):
    """ Not Completed """
    def __init__(self, data, gru_pre_embedding_size=64, gru_hidden_dims=20, num_gru_layers=3, embed_size=32):
        super(GRU4REC, self).__init__()

        self.data = data

        self.num_embeddings = len(self.data.values()[0])
        self.gru_pre_embedding_size = gru_pre_embedding_size
        self.gru_hidden_dims = gru_hidden_dims
        self.num_gru_layers = num_gru_layers
        self.embed_size = embed_size

        self.emb = nn.Embedding(num_embedding=self.num_embeddings, embedding_size=self.gru_pre_embedding_size, sparse=True)
        self.gru = nn.GRU(input_size=self.gru_pre_embedding_size, hidden_size=self.gru_hidden_dims, num_layers=self.num_gru_layers)
    
        self.activation = nn.ReLU()
        self.to_linear = nn.Linear(self.gru_hidden_dims, self.embed_size)

        self.models.apply(self._init_weights)

        self.loss_func = BayesianPersonalizedRankingLoss()

    def _init_weights(self, model):
        if type(model) == nn.Linear:
            nn.init.xavier_normal(model.weight)
            model.bias.data.fill_(0.01)
    
    def _vec_to_device(self, seq, target):
        seq = seq.to(device=self.device, dtype=torch.float)
        target = target.to(device=self.device, dtype=torch.float)
        return seq, target
    
    def _run_epoch(self, seq, target):
        seq = self.emb(seq)
        output, hidden = self.gru(seq, target)

        output = output.view(-1, output.size(-1))
        output = self.activation(self.to_linear(output))

        loss = self.loss_func(output)
        return output, loss 
    
    def training_step(self, batch, batch_idx):
        seq, target = batch 
        seq, target = self._vec_to_device(seq, target)

        total, loss = self._run_epoch(seq, target)
        self.log('training_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        seq, target = batch 
        seq, target = self._vec_to_device(seq, target)

        total, loss = self._run_epoch(seq, target)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), weight_decay=0.0, lr=1e-2)
        return optimizer
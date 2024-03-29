import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class IMG_Encoder(nn.Module):
    def __init__(self, embedding_dim = 4096, dim = 200, margin = None, epsilon = None, dataset=None):
        assert dataset is not None
        super(IMG_Encoder, self).__init__()
        with open('./benchmarks/{}/entity2id.txt'.format(dataset)) as fp:
            entity2id = fp.readlines()[1:]
            entity2id = [i.split('\t')[0] for i in entity2id]
        self.entity2id = entity2id
        self.activation = nn.ReLU()
        self.entity_count = len(entity2id)
        self.dim = dim
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.criterion = nn.MSELoss(reduction='mean') 
        self.raw_embedding = nn.Embedding(self.entity_count, self.dim)
        visual_embs = torch.load("./embeddings/{}-visual.pth".format(dataset))
        self.visual_embedding = nn.Embedding.from_pretrained(visual_embs)

        self.encoder = nn.Sequential(
                torch.nn.Linear(embedding_dim, 192),
                self.activation
            )
        
        self.encoder2 = nn.Sequential(
                torch.nn.Linear(192, self.dim),
                self.activation
            )

        self.decoder2 = nn.Sequential(
                torch.nn.Linear(self.dim, 192),
                self.activation
            )

        self.decoder = nn.Sequential(
                torch.nn.Linear(192, embedding_dim),
                self.activation
            )

    def _init_embedding(self):
        self.ent_embeddings = nn.Embedding(self.entity_count, self.embedding_dim)
        for param in self.ent_embeddings.parameters():
            param.requires_grad = False
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)

    def forward(self, entity_id):
        v1 = self.visual_embedding(entity_id)
        v2 = self.encoder(v1)

        v2_ = self.encoder2(v2)
        v3_ = self.decoder2(v2_)

        v3 = self.decoder(v3_)
        loss = self.criterion(v1, v3)
        return v2_, loss

class TransAE(nn.Module):
    def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None, dataset=None, embedding_dim=None):
        super(TransAE, self).__init__()
        self.dataset = dataset
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.tail_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.ent_embeddings = IMG_Encoder(dim = self.dim, margin = self.margin, epsilon = self.epsilon, dataset=dataset, embedding_dim=embedding_dim)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.tail_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor = self.ent_embeddings.weight.data, 
                a = -self.embedding_range.item(), 
                b = self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor = self.rel_embeddings.weight.data, 
                a= -self.embedding_range.item(), 
                b= self.embedding_range.item()
            )
        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.tail_embeddings(batch_h)
        t = self.tail_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h, t, r, mode)
        if self.margin_flag:
            return self.margin - score, 0
        else:
            return score, 0

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):
        score = self.forward(data)[0]
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()
    
    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
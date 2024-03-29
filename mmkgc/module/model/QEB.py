import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model

import torch.nn.functional as F


class QEB(Model):

    def __init__(
        self,
        ent_tot,
        rel_tot,
        dim=100,
        margin=6.0,
        epsilon=2.0,
        modal_embs=None
    ):

        super(QEB, self).__init__(ent_tot, rel_tot)

        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim
        self.dim_r = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.rel_embeddings_mm = nn.Embedding(self.rel_tot, self.dim_r)

        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )

        self.num_modal = len(modal_embs)
        modal_embeddings = []
        self.mm_dim = 0
        for i in range(self.num_modal):
            modal_embeddings.append(nn.Embedding.from_pretrained(modal_embs[i]).requires_grad_(True))
            dim = modal_embs[i].shape[1]
            self.mm_dim += dim
        self.modal_embeddings = nn.ModuleList(modal_embeddings)
        self.modal_proj = nn.Linear(self.mm_dim, self.dim_e)
        
        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings_mm.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

        self.norm_flag = True
        self.p_norm = 1

    def get_joint_embeddings(self, batch_ent):
        stack = []
        for i in range(self.num_modal):
            stack.append(self.modal_embeddings[i](batch_ent))
        mm_emb = torch.cat(stack, dim=-1)
        return self.modal_proj(mm_emb)


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
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_mm = self.get_joint_embeddings(batch_h)
        t_mm = self.get_joint_embeddings(batch_t)
        r_mm = self.rel_embeddings_mm(batch_r)
        score = self._calc(h, t, r, mode) + self._calc(h_mm, t_mm, r_mm, mode) + self._calc(h_mm, t, r, mode) + self._calc(h_mm, t_mm, r, mode) + self._calc(h_mm, t, r_mm, mode) + self._calc(h, t_mm, r_mm, mode) + self._calc(h, t_mm, r, mode) + self._calc(h, t, r_mm, mode)
        score += self._calc(h + h_mm, t + t_mm, r, mode) + self._calc(h + h_mm, t + t_mm, r_mm, mode)
        score /= 10
        score = self.margin - score
        return score

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

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

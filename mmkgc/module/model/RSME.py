import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class RSME(Model):
    def __init__(self, ent_tot, rel_tot, dim=128, img_dim=768, img_emb=None):
        super(RSME, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.img_dim = img_dim
        self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_re_embeddings = nn.Embedding(self.rel_tot, 2 * self.dim)
        self.rel_im_embeddings = nn.Embedding(self.rel_tot, 2 * self.dim)
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
        self.img_proj = nn.Linear(img_emb.shape[1], 2 * dim)
        self.beta = 0.95

        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
        )
    
    def get_batch_ent_embs(self, data):
        e_re = self.ent_re_embeddings(data)
        e_im = self.ent_re_embeddings(data)
        return torch.cat((e_re, e_im), dim=-1)

    def get_fake_score(
        self,
        batch_h,
        batch_r, 
        batch_t,
        mode,
        fake_hv=None, 
        fake_tv=None
    ):
        if fake_tv is None:
            raise NotImplementedError
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        h_img = self.img_proj(self.img_embeddings(batch_h))
        t_img = fake_tv
        h_re = torch.cat((h_re, h_img[:, 0: self.dim]), dim=-1)
        h_im = torch.cat((h_im, h_img[:, self.dim:]), dim=-1)
        t_re = torch.cat((t_re, t_img[:, 0: self.dim]), dim=-1)
        t_im = torch.cat((t_im, t_img[:, self.dim:]), dim=-1)
        score1 = self._calc(h_re, h_im, t_re, t_im, r_re, r_im)
        score2 = F.cosine_similarity(h_img, t_img, dim=-1)
        score = self.beta * score1 + (1 - self.beta) * score2
        return [score]


    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)

        h_img = self.img_proj(self.img_embeddings(batch_h))
        t_img = self.img_proj(self.img_embeddings(batch_t))
        h_re = torch.cat((h_re, h_img[:, 0: self.dim]), dim=-1)
        h_im = torch.cat((h_im, h_img[:, self.dim:]), dim=-1)
        t_re = torch.cat((t_re, t_img[:, 0: self.dim]), dim=-1)
        t_im = torch.cat((t_im, t_img[:, self.dim:]), dim=-1)
        score1 = self._calc(h_re, h_im, t_re, t_im, r_re, r_im)
        score2 = F.cosine_similarity(h_img, t_img, dim=-1)
        score = self.beta * score1 + (1 - self.beta) * score2
        return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        h_img = self.img_proj(self.img_embeddings(batch_h))
        t_img = self.img_proj(self.img_embeddings(batch_t))
        regul = (torch.mean(h_re ** 2) + 
                 torch.mean(h_im ** 2) + 
                 torch.mean(t_re ** 2) +
                 torch.mean(t_im ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2) + 
                 torch.mean(h_img ** 2) +
                 torch.mean(t_img ** 2)
                 ) / 8
        return regul

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()
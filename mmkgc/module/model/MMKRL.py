import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .Model import Model


class MMKRL(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, p_norm=1, img_emb=None,
                 img_dim=4096, norm_flag=True, margin=None, epsilon=None,
                 text_emb=None):
        super(MMKRL, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.img_dim = img_emb.shape[1]
        self.text_dim = text_emb.shape[1]
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        # 新增的投影矩阵和图像embeddings
        # print(img_emb.shape, text_emb.shape)
        self.mm_proj = nn.Linear(self.img_dim + self.text_dim, self.dim)
        self.mm_embeddings = nn.Embedding.from_pretrained(torch.cat((img_emb, text_emb), dim=1)).requires_grad_(True)
        self.s_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.h_bias = nn.Parameter(torch.randn(self.dim, ), requires_grad=True)
        self.r_bias = nn.Parameter(torch.randn(self.dim, ), requires_grad=True)
        self.t_bias = nn.Parameter(torch.randn(self.dim, ), requires_grad=True)
        self.sm_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.ka_loss = nn.MSELoss()
        
        if margin is None or epsilon is None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if margin is not None:
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
    
    def get_batch_ent_embs(self, data):
        return self.ent_embeddings(data)

    def get_fake_score(
        self,
        batch_h,
        batch_r, 
        batch_t,
        mode,
        fake_hv=None, 
        fake_tv=None
    ):
        if fake_hv is None or fake_tv is None:
            raise NotImplementedError
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_proj = self.s_proj(h) + self.h_bias + fake_hv
        r_proj = self.s_proj(r) + self.r_bias
        t_proj = self.s_proj(t) + self.t_bias + fake_tv
        h_mm_emb = self.mm_proj(self.mm_embeddings(batch_h)) + fake_hv
        t_mm_emb = self.mm_proj(self.mm_embeddings(batch_t)) + fake_tv
        score = (
                self._calc(h_proj, t_proj, r_proj, mode)
                + self._calc(h_mm_emb, t_mm_emb, r_proj, mode)
                + self._calc(h_mm_emb, t, r_proj, mode)
                + self._calc(h, t_mm_emb, r_proj, mode)
        ) / 4
        
        return score


    def forward(self, data, mse=False):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_ent, h_img, t_ent, t_img = batch_h, batch_h, batch_t, batch_t
        mode = data['mode']
        h = self.ent_embeddings(h_ent)
        t = self.ent_embeddings(t_ent)
        r = self.rel_embeddings(batch_r)
        h_proj = self.s_proj(h) + self.h_bias
        r_proj = self.s_proj(r) + self.r_bias
        t_proj = self.s_proj(t) + self.t_bias
        h_mm_emb = self.mm_proj(self.mm_embeddings(batch_h))
        t_mm_emb = self.mm_proj(self.mm_embeddings(batch_t))
        score = (
                self._calc(h_proj, t_proj, r_proj, mode)
                + self._calc(h_mm_emb, t_mm_emb, r_proj, mode)
                + self._calc(h_mm_emb, t, r_proj, mode)
                + self._calc(h, t_mm_emb, r_proj, mode)
        ) / 4
        if not mse:
            return score
        else:
            loss_kas = self.ka_loss(h, h_proj) + self.ka_loss(t, t_proj) + self.ka_loss(r, r_proj)
            loss_kam = self.ka_loss(self.sm_proj(h), h_mm_emb) + self.ka_loss(self.sm_proj(t), t_mm_emb)
            loss_ka = loss_kas + loss_kam
            return score, loss_ka


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
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()

    def set_test_mode(self, new_mode):
        self.test_mode = new_mode


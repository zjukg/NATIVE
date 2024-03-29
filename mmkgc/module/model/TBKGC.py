import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .Model import Model


class TBKGC(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, p_norm=1, img_emb=None,
                 img_dim=4096, norm_flag=True, margin=None, epsilon=None,
                 text_emb=None):
        super(TBKGC, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.img_dim = img_dim
        self.text_dim = text_emb.shape[1]
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        # 新增的投影矩阵和图像embeddings
        self.img_proj = nn.Linear(img_emb.shape[1], self.dim // 2)
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(True)
        self.text_proj = nn.Linear(text_emb.shape[1], self.dim // 2)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)
        
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
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        h_multimodal = torch.cat((h_img_emb, h_text_emb), dim=-1)
        t_multimodal = torch.cat((t_img_emb, t_text_emb), dim=-1)
        # three kinds of fake score
        score_hv = (
                self._calc(h, t, r, mode)
                + self._calc(fake_hv, t_multimodal, r, mode)
                + self._calc(fake_hv, t, r, mode)
                + self._calc(h, t_multimodal, r, mode)
                + self._calc(h + fake_hv, t + t_multimodal, r, mode)
        )
        score_tv = (
                self._calc(h, t, r, mode)
                + self._calc(h_multimodal, fake_tv, r, mode)
                + self._calc(h_multimodal, t, r, mode)
                + self._calc(h, fake_tv, r, mode)
                + self._calc(h + h_multimodal, t + fake_tv, r, mode)
        )
        score_htv = (
                self._calc(h, t, r, mode)
                + self._calc(fake_hv, fake_tv, r, mode)
                + self._calc(fake_hv, t, r, mode)
                + self._calc(h, fake_tv, r, mode)
                + self._calc(h + fake_hv, t + fake_tv, r, mode)
        )
        return [self.margin - score_hv, self.margin - score_tv, self.margin - score_htv], [h_multimodal, t_multimodal]


    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_ent, h_img, t_ent, t_img = batch_h, batch_h, batch_t, batch_t
        mode = data['mode']
        h = self.ent_embeddings(h_ent)
        t = self.ent_embeddings(t_ent)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(h_img))
        t_img_emb = self.img_proj(self.img_embeddings(t_img))
        h_text_emb = self.text_proj(self.text_embeddings(h_img))
        t_text_emb = self.text_proj(self.text_embeddings(t_img))
        h_multimodal = torch.cat((h_img_emb, h_text_emb), dim=-1)
        t_multimodal = torch.cat((t_img_emb, t_text_emb), dim=-1)
        score = (
                self._calc(h, t, r, mode)
                + self._calc(h_multimodal, t_multimodal, r, mode)
                + self._calc(h_multimodal, t, r, mode)
                + self._calc(h, t_multimodal, r, mode)
                + self._calc(h + h_multimodal, t + t_multimodal, r, mode)
        )
        if self.margin_flag:
            return self.margin - score
        else:
            return score


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


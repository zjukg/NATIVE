import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model


class VBTransE(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, p_norm=1, img_emb=None,
                 img_dim=768, norm_flag=True, margin=None, epsilon=None,
                 test_mode='lp', emb_grad=False):
        super(VBTransE, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.img_dim = img_dim
        self.test_mode = test_mode

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        # 新增的投影矩阵和图像embeddings
        self.img_proj = nn.Linear(self.img_dim, self.dim)
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb)
        self.model_type = 'IKRL'
        # 设置img_embedding的的梯度不更新
        if emb_grad == False:
            self.img_embeddings.requires_grad = False

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

    def forward(self, data, batch_size, neg_mode='normal', neg_num=1):
        h_ent, h_img, t_ent, t_img = None, None, None, None
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
        score = (
                self._calc(h, t, r, mode)
                + self._calc(h_img_emb, t_img_emb, r, mode)
                + self._calc(h_img_emb, t, r, mode)
                + self._calc(h, t_img_emb, r, mode)
                # + self._calc(h + h_img_emb, t + t_img_emb, r, mode)
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

    def cross_modal_score_ent2img(self, data):
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
        # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
        score = self._calc(h, t_img_emb, r, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def score_ent2ent(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
        score = self._calc(h, t, r, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def score_vis2vis(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.img_proj(self.img_embeddings(batch_h))
        t = self.img_proj(self.img_embeddings(batch_t))
        r = self.rel_embeddings(batch_r)
        # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
        score = self._calc(h, t, r, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score
    
    def score_vis2ent(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.img_proj(self.img_embeddings(batch_h))
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
        score = self._calc(h, t, r, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score
    
    def score_all2ent(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img = self.img_proj(self.img_embeddings(batch_h))
        # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
        score = self._calc(h, t, r, mode) + self._calc(h_img, t, r, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score
    
    def score_all2vis(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img = self.img_proj(self.img_embeddings(batch_h))
        t_img = self.img_proj(self.img_embeddings(batch_t))
        # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
        score = self._calc(h, t_img, r, mode) + self._calc(h_img, t_img, r, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def predict(self, data):
        if self.test_mode == 'cmlp':
            score = self.cross_modal_score_ent2img(data)
        else:
            score = self.forward(data, batch_size=None, neg_mode='normal')
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()

    def set_test_mode(self, new_mode):
        self.test_mode = new_mode

    def get_rel_rank(self, data):
        head, tail, rel = data
        h_img_emb = self.img_proj(self.img_embeddings(head))
        t_img_emb = self.img_proj(self.img_embeddings(tail))
        relations = self.rel_embeddings.weight
        h = h_img_emb.reshape(-1, h_img_emb.shape[0]).expand((relations.shape[0], h_img_emb.shape[0]))
        t = t_img_emb.reshape(-1, t_img_emb.shape[0]).expand((relations.shape[0], t_img_emb.shape[0]))
        scores = self._calc(h, t, relations, mode='normal')
        ranks = torch.argsort(scores)
        rank = 0
        for (index, val) in enumerate(ranks):
            if val.item() == rel.item():
                rank = index
                break
        return rank + 1

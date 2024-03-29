import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model

class VBRotatE(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, margin=6.0, epsilon=2.0, img_emb=None, img_dim=4096, test_mode='lp'):
        super(VBRotatE, self).__init__(ent_tot, rel_tot)

        self.margin = margin
        self.epsilon = epsilon

        self.dim_e = dim * 2
        self.dim_r = dim

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.img_dim = img_dim
        self.img_proj = nn.Linear(self.img_dim, self.dim_e)
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb)
        self.test_mode = test_mode

        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), 
            requires_grad=False
        )
        

        nn.init.uniform_(
            tensor = self.ent_embeddings.weight.data, 
            a=-self.ent_embedding_range.item(), 
            b=self.ent_embedding_range.item()
        )

        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), 
            requires_grad=False
        )

        nn.init.uniform_(
            tensor = self.rel_embeddings.weight.data, 
            a=-self.rel_embedding_range.item(), 
            b=self.rel_embedding_range.item()
        )

        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

    def _calc(self, h, t, r, mode):
        pi = self.pi_const

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

        if mode == "head_batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0).sum(dim = -1)
        return score.permute(1, 0).flatten()

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
        # print(h.shape, t.shape, r.shape, h_img_emb.shape, t_img_emb.shape)
        score = (
                self._calc(h, t, r, mode)
                + self._calc(h_img_emb, t_img_emb, r, mode)
                + self._calc(h_img_emb, t, r, mode)
                + self._calc(h, t_img_emb, r, mode)
        )
        score = self.margin - score
        return score
    
    def cross_modal_score_ent2img(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_ent, h_img, t_ent, t_img = batch_h, batch_h, batch_t, batch_t
        mode = data['mode']
        h = self.ent_embeddings(h_ent)
        r = self.rel_embeddings(batch_r)
        t_img_emb = self.img_proj(self.img_embeddings(t_img))
        # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
        score = self._calc(h, t_img_emb, r, mode)
        score = self.margin - score
        return score

    def predict(self, data):
        if self.test_mode == "cmlp":
            score = -self.cross_modal_score_ent2img(data)
        else:
            score = -self.forward(data, batch_size=1, neg_mode='normal')
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

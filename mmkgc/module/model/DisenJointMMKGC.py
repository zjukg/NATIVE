import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, y_dim)
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, y_dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /2./logvar.exp()).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.0

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class DisenJointMMKGC(Model):

    def __init__(
        self,
        ent_tot,
        rel_tot,
        dim=100,
        margin=6.0,
        epsilon=2.0,
        img_emb=None,
        text_emb=None
    ):

        super(DisenJointMMKGC, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None
        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim * 2
        self.dim_r = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )
        self.img_dim = img_emb.shape[1]
        self.text_dim = text_emb.shape[1]
        self.img_proj = nn.Linear(self.img_dim, self.dim_e)
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(True)
        self.text_proj = nn.Linear(self.text_dim, self.dim_e)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)

        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
        self.ent_attn.requires_grad_(True)
        self.rel_gate = nn.Embedding(self.rel_tot, self.dim_e)
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
            tensor=self.rel_gate.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
        self.ent_attn.requires_grad_(True)
        # Disentangle Module for Three Modalities
        self.disen_modules = nn.ModuleList([
            # s -> v + t
            CLUBSample(2 * self.dim_e, self.dim_e, 2 * self.dim_e),
            # v -> t
            CLUBSample(self.dim_e, self.dim_e, self.dim_e),
        ])
        self.batch_size = 1024
        self.rel_embeddings_s = nn.Embedding(self.rel_tot, self.dim_r)
        self.rel_embeddings_v = nn.Embedding(self.rel_tot, self.dim_r)
        self.rel_embeddings_t = nn.Embedding(self.rel_tot, self.dim_r)
        nn.init.uniform_(
            tensor=self.rel_embeddings_s.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings_v.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings_t.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )

    
    def gated_fusion(self, emb, rel):
        # emb: batch_size x dim
        # rel: batch_size x dim
        w = torch.sigmoid(emb * rel)
        return w * emb + (1 - w) * rel
        

    def get_joint_embeddings(self, es, ev, et, rg):
        e = torch.stack((es, ev, et), dim=1)
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=-1)
        context_vectors = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)
        return context_vectors, attention_weights
    
    def cal_score(self, embs):
        return self._calc(embs[0], embs[2], embs[1], "")

    def _calc(self, h, t, r, mode):
        pi = self.pi_const

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1,
                               re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1,
                               re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1,
                               re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1,
                               re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(
            -1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(
            -1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

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

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.permute(1, 0).flatten()

    def get_disentangle_loss(self, es, ev, et):
        es = es[0:self.batch_size]
        ev = es[0:self.batch_size]
        et = es[0:self.batch_size]
        disen_loss = 0
        disen_loss += self.disen_modules[0](torch.cat((ev, et), dim=-1), es)
        disen_loss += self.disen_modules[1](ev, et)
        return disen_loss / 2.0
    
    def train_disen_module(self, data):
        batch_h = data['batch_h'][:self.batch_size]
        h = self.ent_embeddings(batch_h)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        mm_input = torch.cat((h_img_emb, h_text_emb), dim=-1)
        learning_loss = 0
        learning_loss += self.disen_modules[0].learning_loss(mm_input, h)
        learning_loss += self.disen_modules[1].learning_loss(h_img_emb, h_text_emb)
        return learning_loss / 2.0

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        rs = self.rel_embeddings_s(batch_r)
        rv = self.rel_embeddings_v(batch_r)
        rt = self.rel_embeddings_v(batch_r)
        rg = self.rel_gate(batch_r)
        h_joint, w1 = self.get_joint_embeddings(h, h_img_emb, h_text_emb, rg)
        t_joint, w2 = self.get_joint_embeddings(t, t_img_emb, t_text_emb, rg)
        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        disen_loss = self.get_disentangle_loss(h, h_img_emb, h_text_emb)
        score_s = self.margin - self._calc(h, t, rs, mode)
        score_v = self.margin - self._calc(h_img_emb, t_img_emb, rv, mode)
        score_t = self.margin - self._calc(h_text_emb, t_text_emb, rt, mode)
        return score, disen_loss, [score_s, score_t, score_v], (w1 + w2).detach() / 2

    def predict(self, data):
        score, _, scores, weights = self.forward(data)
        for i in range(3):
            score += scores[i] * weights[:, i]
        return -score.cpu().data.numpy()

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
    

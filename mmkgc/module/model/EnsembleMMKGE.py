import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class EnsembleMMKGE(Model):

	def __init__(
		self,
		ent_tot, 
		rel_tot, 
		dim = 100, 
		margin = 6.0, 
		epsilon = 2.0,
		visual_embs=None,
		textual_embs=None
	):
		super(EnsembleMMKGE, self).__init__(ent_tot, rel_tot)

		self.margin = margin
		self.epsilon = epsilon

		self.dim_e = dim * 2
		self.dim_r = dim

		self.ent_emb_s = nn.Embedding(self.ent_tot, self.dim_e)
		self.rel_emb_s = nn.Embedding(self.rel_tot, self.dim_r)
		
		self.ent_emb_v = nn.Embedding.from_pretrained(visual_embs)
		self.ent_emb_t = nn.Embedding.from_pretrained(textual_embs)
		self.ent_emb_v.requires_grad_(True)
		self.ent_emb_t.requires_grad_(True)
		visual_dim = self.ent_emb_v.weight.shape[1]
		textual_dim = self.ent_emb_t.weight.shape[1]
		self.visual_proj = nn.Linear(visual_dim, self.dim_e)
		self.textual_proj = nn.Linear(textual_dim, self.dim_e)
		self.rel_emb_v = nn.Embedding(self.rel_tot, self.dim_r)
		self.rel_emb_t = nn.Embedding(self.rel_tot, self.dim_r)
		self.rel_emb_j = nn.Embedding(self.rel_tot, self.dim_r)
		self.init_emb()
		self.predict_mode = "all"
		self.ent_attn = nn.Parameter(torch.zeros((self.dim_e, )))
		self.ent_attn.requires_grad_(True)


	def init_emb(self):
		self.ent_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), 
			requires_grad=False
		)
		nn.init.uniform_(
			tensor = self.ent_emb_s.weight.data, 
			a=-self.ent_embedding_range.item(), 
			b=self.ent_embedding_range.item()
		)
		self.rel_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), 
			requires_grad=False
		)
		nn.init.uniform_(
			tensor = self.rel_emb_s.weight.data, 
			a=-self.rel_embedding_range.item(), 
			b=self.rel_embedding_range.item()
		)
		nn.init.uniform_(
			tensor = self.rel_emb_v.weight.data, 
			a=-self.rel_embedding_range.item(), 
			b=self.rel_embedding_range.item()
		)
		nn.init.uniform_(
			tensor = self.rel_emb_t.weight.data, 
			a=-self.rel_embedding_range.item(), 
			b=self.rel_embedding_range.item()
		)
		nn.init.uniform_(
			tensor = self.rel_emb_j.weight.data, 
			a=-self.rel_embedding_range.item(), 
			b=self.rel_embedding_range.item()
		)
		self.margin = nn.Parameter(torch.Tensor([self.margin]))
		self.margin.requires_grad = False


	def score_function_transe(self, h, r, t, mode):
		h = F.normalize(h, 2, -1)
		r = F.normalize(r, 2, -1)
		t = F.normalize(t, 2, -1)
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, 1, -1).flatten()
		return score


	def score_function_rotate(self, h, t, r, mode):
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

	def forward(self, data, require_att=False):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		# structural embeddings
		h = self.ent_emb_s(batch_h)
		t = self.ent_emb_s(batch_t)
		r = self.rel_emb_s(batch_r)
		# visual embeddings
		hv = self.visual_proj(self.ent_emb_v(batch_h))
		tv = self.visual_proj(self.ent_emb_v(batch_t))
		rv = self.rel_emb_v(batch_r)
		# textual embeddings
		ht = self.textual_proj(self.ent_emb_t(batch_h))
		tt = self.textual_proj(self.ent_emb_t(batch_t))
		rt = self.rel_emb_t(batch_r)
		# joint embeddings
		hj, att_h = self.get_joint_embeddings(h, hv, ht)
		tj, att_t = self.get_joint_embeddings(t, tv, tt)
		rj = self.rel_emb_j(batch_r)
		# scores
		score_s = self.margin - self.score_function_rotate(h, t, r, mode)
		score_v = self.margin - self.score_function_rotate(hv, tv, rv, mode)
		score_t = self.margin - self.score_function_rotate(ht, tt, rt, mode)
		score_j = self.margin - self.score_function_rotate(hj, tj, rj, mode)
		if require_att:
			return [score_s, score_v, score_t, score_j], (att_h + att_t)
		return [score_s, score_v, score_t, score_j]
	
	def get_joint_embeddings(self, es, ev, et):
		e = torch.stack((es, ev, et), dim=1)
		dot = torch.exp(e @ self.ent_attn)
		att_w = dot / torch.sum(dot, dim=1).reshape(-1, 1)
		w1, w2, w3 = att_w[:, 0].reshape(-1, 1), att_w[:, 1].reshape(-1, 1), att_w[:, 2].reshape(-1, 1)
		ej = w1 * es + w2 * ev + w3 * et
		return ej, att_w
	

	def predict(self, data):
		pred_result, att = self.forward(data, require_att=True)
		if self.predict_mode == "s":
			score = -pred_result[0]
		elif self.predict_mode == "v":
			score = -pred_result[1]
		elif self.predict_mode == "t":
			score = -pred_result[2]
		elif self.predict_mode == "j":
			score = -pred_result[3]
		elif self.predict_mode == "all":
			att /= 2
			w1, w2, w3 = att[:, 0].reshape(-1, 1), att[:, 1].reshape(-1, 1), att[:, 2].reshape(-1, 1)
			score = -(w1 * pred_result[0] + w2 * pred_result[1] + w3 * pred_result[2] + pred_result[3])
		else:
			raise NotImplementedError("No such prediction setting!")
		return score.cpu().data.numpy()

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_emb_s(batch_h)
		t = self.ent_emb_s(batch_t)
		r = self.rel_emb_s(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

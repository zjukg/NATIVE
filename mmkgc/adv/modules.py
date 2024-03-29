import torch
import torch.nn as nn


class BaseGenerator(nn.Module):
    def __init__(
        self, 
        noise_dim, 
        structure_dim, 
        img_dim
    ):
        super(BaseGenerator, self).__init__()
        self.proj_dim = 512
        self.noise_dim = noise_dim
        self.generator_model = nn.Sequential(
            nn.Linear(noise_dim + structure_dim, self.proj_dim),
            nn.LeakyReLU(),
            nn.Linear(self.proj_dim, img_dim)
        )


    def forward(self, batch_ent_emb):
        random_noise = torch.randn((batch_ent_emb.shape[0], self.noise_dim)).cuda()
        batch_data = torch.cat((random_noise, batch_ent_emb), dim=-1)
        out = self.generator_model(batch_data)
        return out


class RandomGenerator(nn.Module):
    def __init__(
        self, 
        noise_dim,
        img_dim
    ):
        super(RandomGenerator, self).__init__()
        self.proj_dim = 256
        self.noise_dim = noise_dim
        self.generator_model = nn.Sequential(
            nn.Linear(noise_dim, self.proj_dim),
            nn.LeakyReLU(),
            nn.Linear(self.proj_dim, img_dim)
        )


    def forward(self, batch_ent_emb):
        random_noise = torch.randn((batch_ent_emb.shape[0], self.noise_dim)).cuda()
        out = self.generator_model(random_noise)
        return out


class MultiGenerator(nn.Module):
    def __init__(
        self, 
        noise_dim, 
        structure_dim, 
        img_dim
    ):
        super(MultiGenerator, self).__init__()
        self.img_generator = BaseGenerator(noise_dim, structure_dim, img_dim)
        self.text_generator = BaseGenerator(noise_dim, structure_dim, img_dim)
    
    def forward(self, batch_ent_emb, modal):
        if modal == 1:
            return self.img_generator(batch_ent_emb)
        elif modal == 2:
            return self.text_generator(batch_ent_emb)
        else:
            raise NotImplementedError


class CombinedGenerator(nn.Module):
    def __init__(
        self, 
        noise_dim, 
        structure_dim, 
        img_dim
    ):
        super(CombinedGenerator, self).__init__()
        self.generator = BaseGenerator(noise_dim, structure_dim * 3, img_dim * 2)
    
    def forward(
        self,
        batch_ent_s,
        batch_ent_v,
        batch_ent_t
    ):
        batch_ent_emb = torch.cat((batch_ent_s, batch_ent_v, batch_ent_t), dim=-1)
        batch_gen = self.generator(batch_ent_emb)
        batch_gen_v, batch_gen_t = torch.chunk(batch_gen, chunks=2, dim=-1)
        return batch_gen_v, batch_gen_t

class CombinedGenerator2(nn.Module):
    def __init__(
        self, 
        noise_dim, 
        structure_dim, 
        img_dim
    ):
        super(CombinedGenerator2, self).__init__()
        self.generator = BaseGenerator(noise_dim, structure_dim * 5, img_dim * 2)
    
    def forward(
        self,
        batch_ent_s,
        batch_ent_v,
        batch_ent_t,
        batch_ent_a,
        batch_ent_video
    ):
        batch_ent_emb = torch.cat((batch_ent_s, batch_ent_v, batch_ent_t, batch_ent_a, batch_ent_video), dim=-1)
        batch_gen = self.generator(batch_ent_emb)
        batch_gen_i, batch_gen_t, batch_gen_a, batch_gen_v = torch.chunk(batch_gen, chunks=4, dim=-1)
        return batch_gen_i, batch_gen_t, batch_gen_a, batch_gen_v

class CombinedGenerator3(nn.Module):
    def __init__(
        self, 
        noise_dim, 
        structure_dim, 
        img_dim
    ):
        super(CombinedGenerator3, self).__init__()
        self.generator = BaseGenerator(noise_dim, structure_dim * 4, img_dim * 2)
    
    def forward(
        self,
        batch_ent_s,
        batch_ent_v,
        batch_ent_t,
        batch_ent_n
    ):
        batch_ent_emb = torch.cat((batch_ent_s, batch_ent_v, batch_ent_t, batch_ent_n), dim=-1)
        batch_gen = self.generator(batch_ent_emb)
        batch_gen_i, batch_gen_t, batch_gen_a = torch.chunk(batch_gen, chunks=3, dim=-1)
        return batch_gen_i, batch_gen_t, batch_gen_a


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp



class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.sim_func = Similarity(temp=temp)

    def forward(self, node_emb, img_emb):
        batch_sim = self.sim_func(node_emb.unsqueeze(1), img_emb.unsqueeze(0))
        labels = torch.arange(batch_sim.size(0)).long().to('cuda')
        return self.loss(batch_sim, labels)
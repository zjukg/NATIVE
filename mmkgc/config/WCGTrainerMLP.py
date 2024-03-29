# coding:utf-8
from calendar import c
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm


class WCGTrainerMLP(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_times=1000,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="sgd",
                 save_steps=None,
                 checkpoint_dir=None,
                 generator=None,
                 lrg=None,
                 mu=None):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha
        # learning rate of the generator
        assert lrg is not None
        self.alpha_g = lrg

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        
        # the generator part
        assert generator is not None
        assert mu is not None
        self.optimizer_g = None
        self.generator = generator
        self.batch_size = self.model.batch_size
        self.generator.cuda()
        self.mu = mu

    def train_one_step(self, data):
        # training D
        self.optimizer.zero_grad()
        loss, _, real_embs = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })
        real_embs = [
            real_embs[0][:self.batch_size],
            real_embs[1][:self.batch_size],
            real_embs[2][:self.batch_size]
        ]
        # generate fake multimodal feature
        batch_h_gen = self.to_var(data['batch_h'][0: self.batch_size], self.use_gpu)
        batch_t_gen = self.to_var(data['batch_t'][0: self.batch_size], self.use_gpu)
        batch_r = self.to_var(data['batch_r'][0: self.batch_size], self.use_gpu)
        batch_hs, batch_hv, batch_ht = self.model.model.get_batch_ent_multimodal_embs(batch_h_gen)
        batch_ts, batch_tv, batch_tt = self.model.model.get_batch_ent_multimodal_embs(batch_t_gen)
        batch_gen_hv, batch_gen_ht = self.generator(batch_hs, batch_hv, batch_ht)
        batch_gen_tv, batch_gen_tt = self.generator(batch_ts, batch_tv, batch_tt)
        
        real_h_in = (batch_hs + batch_hv + batch_ht) / 3
        real_t_in = (batch_ts + batch_tv + batch_tt) / 3
        real_score_h = self.model.model.adv_scores(real_h_in)
        real_score_t = self.model.model.adv_scores(real_t_in)
        gen_h_in = (batch_hs + batch_gen_hv + batch_gen_ht) / 3
        gen_t_in = (batch_ts + batch_gen_tv + batch_gen_tt) / 3
        gen_score_h = self.model.model.adv_scores(gen_h_in)
        gen_score_t = self.model.model.adv_scores(gen_t_in)
        # when training D: positive_score > fake_score
        #### chang this loss
        loss += self.mu * torch.mean((-real_score_h + gen_score_h) + (-real_score_t + gen_score_t))
        loss += self.mu * (self.calc_gradient_penalty(real_embs[0], gen_score_h) + self.calc_gradient_penalty(real_embs[2], gen_score_t))
        loss.backward()
        self.optimizer.step()
        # training G
        self.optimizer_g.zero_grad()
        batch_hs, batch_hv, batch_ht = self.model.model.get_batch_ent_multimodal_embs(batch_h_gen)
        batch_ts, batch_tv, batch_tt = self.model.model.get_batch_ent_multimodal_embs(batch_t_gen)
        batch_gen_hv, batch_gen_ht = self.generator(batch_hs, batch_hv, batch_ht)
        batch_gen_tv, batch_gen_tt = self.generator(batch_ts, batch_tv, batch_tt)
        gen_h_in = (batch_hs + batch_gen_hv + batch_gen_ht) / 3
        gen_t_in = (batch_ts + batch_gen_tv + batch_gen_tt) / 3
        gen_score_h = self.model.model.adv_scores(gen_h_in)
        gen_score_t = self.model.model.adv_scores(gen_t_in)
        loss_g = 0.0
        #### chang this loss
        loss_g += -torch.mean(gen_score_h + gen_score_t)
        loss_g.backward()
        # self.optimizer_g.step()
        return loss.item(), loss_g.item()
    
    def calc_gradient_penalty(self, real_data, fake_data):
        batchsize = real_data.shape[0]
        alpha = torch.rand(batchsize, 1).cuda()
        inter_h = alpha * real_data + ((1 - alpha) * fake_data)
        inter_h = torch.autograd.Variable(inter_h, requires_grad=True)
        scores = self.model.model.adv_scores(inter_h)

        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=inter_h,
            grad_outputs=torch.ones(scores.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 0.1 #opt.GP_LAMBDA
        return gradient_penalty

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )
            print(
                "Learning Rate of D: {}\nLearning Rate of G: {}".format(
                    self.alpha, self.alpha_g)
            )
        else:
            raise NotImplementedError
        print("Finish initializing...")

        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            res_g = 0.0
            for data in self.data_loader:
                loss, loss_g = self.train_one_step(data)
                res += loss
                res_g += loss_g
            training_range.set_description("Epoch %d | D loss: %f, G loss %f" % (epoch, res, res_g))

            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir



# coding:utf-8
import torch
import torch.nn as nn
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


class DisenAdvTrainer(object):

    def __init__(self,
        model=None,
        data_loader=None,
        train_times=1000,
        alpha=0.5,
        use_gpu=True,
        opt_method="sgd",
        save_steps=None,
        checkpoint_dir=None,
        train_mode='adp',
        alpha2=0.0001,
        generator=None,
        lrg=None,
        mu=None
    ):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.optimizer_dis = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha
        self.alpha2 = alpha2
        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir

        self.train_mode = train_mode

        assert lrg is not None
        self.alpha3 = lrg
        # the generator part
        assert generator is not None
        assert mu is not None
        self.optimizer_gen = None
        self.generator = generator
        self.batch_size = self.model.batch_size
        self.generator.cuda()
        self.mu = mu
        self.count = 0


    def train_one_step(self, data):
        # Train KGE model
        self.optimizer.zero_grad()
        data_input = {
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        }
        loss, p_score = self.model(data_input)
        # generate fake multimodal feature
        batch_h_gen = self.to_var(data['batch_h'][0: self.batch_size], self.use_gpu)
        batch_t_gen = self.to_var(data['batch_t'][0: self.batch_size], self.use_gpu)
        batch_r = self.to_var(data['batch_r'][0: self.batch_size], self.use_gpu)
        batch_hs, batch_hv, batch_ht = self.model.model.get_batch_ent_multimodal_embs(batch_h_gen, batch_r)
        batch_ts, batch_tv, batch_tt = self.model.model.get_batch_ent_multimodal_embs(batch_t_gen, batch_r)
        batch_gen_hv, batch_gen_ht = self.generator(batch_hs, batch_hv, batch_ht)
        batch_gen_tv, batch_gen_tt = self.generator(batch_ts, batch_tv, batch_tt)
        scores, _ = self.model.model.get_fake_score(
            batch_h=batch_h_gen,
            batch_r=batch_r,
            batch_t=batch_t_gen,
            mode=data['mode'],
            fake_hv=batch_gen_hv,
            fake_tv=batch_gen_tv,
            fake_ht=batch_gen_ht,
            fake_tt=batch_gen_tt
        )
        for score in scores:
            loss += self.mu * self.model.loss(p_score, score)
        loss.backward()
        self.optimizer.step()
        disen_loss = 0.0
        loss_g = 0.0
        if self.count % 2:
            # Train MI Estimator
            self.optimizer_dis.zero_grad()
            disen_loss = self.model.model.train_disen_module(data_input)
            disen_loss.backward()
            self.optimizer_dis.step()
        else:
            # Train Generator
            self.optimizer_gen.zero_grad()
            batch_hs, batch_hv, batch_ht = self.model.model.get_batch_ent_multimodal_embs(batch_h_gen, batch_r)
            batch_ts, batch_tv, batch_tt = self.model.model.get_batch_ent_multimodal_embs(batch_t_gen, batch_r)
            batch_gen_hv, batch_gen_ht = self.generator(batch_hs, batch_hv, batch_ht)
            batch_gen_tv, batch_gen_tt = self.generator(batch_ts, batch_tv, batch_tt)
            p_score = self.model({
                'batch_h': batch_h_gen,
                'batch_t': batch_t_gen,
                'batch_r': batch_r,
                'batch_y': self.to_var(data['batch_y'], self.use_gpu),
                'mode': data['mode']
            }, fast_return=True)

            scores, _ = self.model.model.get_fake_score(
                batch_h=batch_h_gen,
                batch_r=batch_r,
                batch_t=batch_t_gen,
                mode=data['mode'],
                fake_hv=batch_gen_hv,
                fake_tv=batch_gen_tv,
                fake_ht=batch_gen_ht,
                fake_tt=batch_gen_tt
            )
            
            for score in scores:
                loss_g += self.model.loss(score, p_score) / 3
            loss_g.backward()
            self.optimizer_gen.step()
        return loss.item(), disen_loss.item() if disen_loss != 0.0 else 0, loss_g.item() if loss_g != 0.0 else 0

    def run(self):
        if self.use_gpu:
            self.model.cuda()
        mi_disc_params = list(map(id, self.model.model.disen_modules.parameters()))
        rest_params = filter(lambda x:id(x) not in mi_disc_params, self.model.model.parameters())
        self.optimizer = optim.Adam(
            rest_params,
            lr=self.alpha,
            weight_decay=self.weight_decay,
        )
        self.optimizer_dis = optim.Adam(
            self.model.model.disen_modules.parameters(),
            lr=self.alpha2,
            weight_decay=self.weight_decay,
        )
        self.optimizer_gen = optim.Adam(
            self.generator.parameters(),
            lr=self.alpha3,
            weight_decay=self.weight_decay,
        )
        print("Finish initializing...")

        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            res_dis = 0.0
            res_gen = 0.0
            for data in self.data_loader:
                loss, loss_dis, loss_gen = self.train_one_step(data)
                res += loss
                res_dis += loss_dis
                res_gen += loss_gen
            self.count += 1
            training_range.set_description("Epoch %d | loss: %f loss_dis: %f loss_gen: %f" % (epoch, res, res_dis, res_gen))

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

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


class DisenTrainer(object):

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
                 alpha2=0.0001):

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


    def train_one_step(self, data):
        self.optimizer.zero_grad()
        data = {
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        }
        loss, _ = self.model(data)
        loss.backward()
        self.optimizer.step()
        self.optimizer_dis.zero_grad()
        disen_loss = self.model.model.train_disen_module(data)
        disen_loss.backward()
        self.optimizer_dis.step()
        return loss.item(), disen_loss.item()

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
        print("Finish initializing...")

        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            res_dis = 0.0
            for data in self.data_loader:
                loss, loss_dis = self.train_one_step(data)
                res += loss
                res_dis += loss_dis
            training_range.set_description("Epoch %d | loss: %f loss_dis: %f" % (epoch, res, res_dis))

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

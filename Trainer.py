# coding:utf-8
import torch
from torch.autograd import Variable
import torch.optim as optim
import os
import numpy as np
import time

class Trainer(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_times=1000,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="sgd",
                 save_steps=None,
                 checkpoint_dir=None,
                 early_stopping_enabled=True):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_enabled = early_stopping_enabled

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        self.model.startingBatch()
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")

        prev_losses = []
        for epoch in range(self.train_times):
            res = 0.0
            start = time.perf_counter()
            start_neg = time.perf_counter()
            time_neg = 0
            for data in self.data_loader:
                end_neg = time.perf_counter()
                time_neg+=end_neg-start_neg
                loss = self.train_one_step(data)
                res += loss
                start_neg = time.perf_counter()
            end = time.perf_counter()
            print("Epoch:",epoch,"; Loss:",res,"; Time:", end-start,"; Time neg.:",time_neg)
            prev_losses.append(res)

            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + ".ckpt"))

                # Early stopping: the train loss is less than 1e-2 or the train loss in the last steps was stable.
                std = np.std(np.array(prev_losses))
                print("Epoch %d has finished, saving..." % (epoch), "; Std. dev. of  prev steps:", std)
                if self.early_stopping_enabled and (res < 0.01 or std < 0.1):
                    break
                prev_losses = []

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
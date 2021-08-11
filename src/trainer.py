import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from loader import  get_dataloader
from models import MLP, MLPLinear
from ogb.nodeproppred import Evaluator
import glob
from utils import to_device, prepare_folder, seed_everything
from loss import get_entropy, get_cond_entropy, get_kld
from logger import Logger
from torch.utils.data import TensorDataset, DataLoader

"""
Livia Model
"""
class LTrainer():

    def __init__(self, args):
        self.args = args
        self.device = self._device()
        print(f"Using {self.device}")
        self.basic_loss_tim = 0
        self.entropy_tim = 0
        self.conditional_entropy_tim = 0
        self.kld = 0
        self._build()

    def _build(self):
        print(self.args)
        self._build_loaders()
        self._build_model()
        self._build_criteria_and_optim()
        self._build_scheduler()
        if self.args.tim:
            self.pi = torch.zeros(self.num_classes).to(self.device)
            for elem in self.y_true.squeeze(1)[self.split_idx['test']]:
              self.pi[elem]+=1.0
            self.pi = self.pi/len(self.y_true.squeeze(1)[self.split_idx['test']])
            self.train()

    def _build_loaders(self):
        if self.args.dataset in ["arxiv", "products"]:
            self.x, self.y_true, self.train_idx, self.num_classes, self.split_idx = get_dataloader(self.args, self.device)
            self.test_dataset = DataLoader(TensorDataset(self.x[self.split_idx['test']],self.y_true[self.split_idx['test']]), batch_size = 40000)

    def _build_model(self):
        if self.args.model == "mlp":
            self.model: nn.Module = to_device(MLP(in_channels=self.x.size(-1),out_channels=self.num_classes, relu_first=(self.args.dataset == 'products')).cuda(), self.device)

    def _build_criteria_and_optim(self):
        if self.args.criterion == "ce":
            self.criterion = nn.CrossEntropyLoss()
        elif self.args.criterion == "nll":
            self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.param eters(), lr=self.args.lr)
        self.model_dir = prepare_folder(self.args.exp_name, self.model)
        self.evaluator = Evaluator(name=f'ogbn-{self.args.dataset}')

    def _build_scheduler(self):
        rate_decay_step_size = 40
        rate_decay_gamma = 0.8
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=rate_decay_step_size, gamma=rate_decay_gamma)

    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_epoch(self,epoch):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.x[self.train_idx])
        loss = self.criterion(out, self.y_true.squeeze(1)[self.train_idx])
        loss.backward()
        self.optimizer.step()
        return loss.item(), 0

    def train_epoch_adapt(self,epoch):
        self.model.train()
        self.optimizer.zero_grad()
        for batchid, data in enumerate(self.test_dataset):
            x,y = data
            prob=self.model(x)
            kl_divergence = get_kld(self.pi, prob)
            conditional_entropy = get_cond_entropy(prob)
            self.conditional_entropy_tim = conditional_entropy.item()
            self.kld = kl_divergence.item()
            loss2 = (self.args.alpha *conditional_entropy) + (self.args.gamma * kl_divergence)
            loss2.backward()
            for i,p in enumerate(self.model.lins[-1].parameters()):
                    p.data = p.data - self.args.lr_tim * p.grad
        return 0,loss2.item()

    def train(self):
        logger = Logger(self.args.runs, self.args)
        for run in range(self.args.runs):
            self._build_model()
            self._build_criteria_and_optim()
            self._build_scheduler()
            import gc
            gc.collect()
            seed_everything(run)
            self.model.reset_parameters()
            best_valid = 0
            best_out = None
            final = []

            for epoch in range(0, self.args.epochs):
                loss,loss_tim = self.train_epoch(epoch)
                result, out = self.test_epoch()
                train_acc, valid_acc, test_acc = result
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    best_out = out.cpu().exp()
                    best_result = result

            for epoch in range(0, 100):
                loss, loss_tim = self.train_epoch_adapt(epoch)
                result, out = self.test_epoch()
                train_acc, valid_acc, test_acc = result
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    best_out = out.cpu().exp()
                    best_result = result


            logger.add_result(run, best_result)

            torch.save(best_out, f'../configs/{self.args.exp_name}/{run}.pt')
            # self.plot_losses(final)
        logger.print_statistics()


    def plot_losses(self,final):
        total_tim_loss = [elem[0] for elem in final]
        total_kld = [elem[1] for elem in final]
        total_cond_ent = [elem[2] for elem in final]
        total_cross_ent = [elem[3] for elem in final]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([i for i in range(400)], total_tim_loss, color='orange', label ='Total TIM Loss')
        ax.plot([i for i in range(400)], total_kld, color='blue', label = 'KL Divergence')
        ax.plot([i for i in range(400)], total_cond_ent, color='green', label = 'Conditional Entropy')
        ax.plot([i for i in range(400)], total_cross_ent, color='red', label = 'Cross Entropy')
        pself.legend(["Total TIM Loss", "KL Divergence", "Conditional Entropy","Cross Entropy"], loc='best')
        fig.suptitle('Loss', fontsize=20)
        pself.xlabel('Epochs', fontsize=18)
        fig.savefig(f'../plots/{self.args.exp_name}Loss.png')


    @torch.no_grad()
    def test_epoch(self):
        self.model.eval()

        out = self.model(self.x)
        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = self.evaluator.eval({
            'y_true': self.y_true[self.split_idx['train']],
            'y_pred': y_pred[self.split_idx['train']],
        })['acc']
        valid_acc = self.evaluator.eval({
            'y_true': self.y_true[self.split_idx['valid']],
            'y_pred': y_pred[self.split_idx['valid']],
        })['acc']
        test_acc = self.evaluator.eval({
            'y_true': self.y_true[self.split_idx['test']],
            'y_pred': y_pred[self.split_idx['test']],
        })['acc']

        return (train_acc, valid_acc, test_acc), out

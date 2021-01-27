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
import wandb

"""
Livia_MAML Trainer
"""
class La_MAML_Trainer():

    def __init__(self, args):
        self.args = args
        wandb.init(project="TIM_MAML", name=f"{args.exp_name}", config=args)
        wandb.config = args
        self.device = self._device()
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
            self.pi = torch.zeros(40).to(self.device)
            for elem in self.y_true.squeeze(1)[self.split_idx['valid']]:
              self.pi[elem]+=1.0
            self.pi = self.pi/len(self.y_true.squeeze(1)[self.split_idx['valid']])
        self.train()

    def _build_loaders(self):
        if self.args.dataset in ["arxiv", "products"]:
            self.x, self.y_true, self.train_idx, self.num_classes, self.split_idx = get_dataloader(self.args, self.device)

    def _build_model(self):
        if self.args.model == "mlp":
            self.model: nn.Module = to_device(MLP(in_channels=self.x.size(-1),out_channels=self.num_classes, relu_first=(self.args.dataset == 'products')).cuda(), self.device)

    def _build_criteria_and_optim(self):
        if self.args.criterion == "ce":
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer_inner = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.model_dir = prepare_folder(self.args.exp_name, self.model)
        self.evaluator = Evaluator(name=f'ogbn-{self.args.dataset}')

    def _build_scheduler(self):
        rate_decay_step_size = 40
        rate_decay_gamma = 0.8
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer_inner, step_size=rate_decay_step_size, gamma=rate_decay_gamma)

    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_model_grads(self):
        out = self.model(self.x[self.train_idx])
        loss = self.criterion(out, self.y_true.squeeze(1)[self.train_idx])
        loss.backward()
        outer_grads = {}
        for i,p in enumerate(self.model.lins[-1].parameters()):
            outer_grads[i] = p.grad
        return outer_grads

    def weight_initialization(self):
        for i,p in enumerate(self.model.parameters()):
            if i==1:
              for j in range(40):
                p.data[j] = self.pi[j]

    def train(self):
        logger = Logger(self.args.runs, self.args)
        for self.run in range(self.args.runs):
            import gc
            gc.collect()
            seed_everything(self.run)
            self.model.reset_parameters()
            best_valid = 0
            best_out = None
            self.weight_initialization()
            for epoch in range(1, self.args.epochs):

                loss_main, loss_tim = self.train_epoch(epoch)
                # loss_outer = self.train_outer_step()
                result, out = self.test_epoch()
                train_acc, valid_acc, test_acc = result
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    best_out = out.cpu().exp()
                if not self.run and epoch%20:
                    wandb.log({f"Loss_TIM_Maml": loss_tim,
                               f"Train": train_acc,
                               f"Valid": valid_acc,
                               f"Test": test_acc,
                               f"Basic_loss_tim": loss_main,
                               f"KL_Divergence": self.kld,
                               f"Conditional_entropy_tim": self.conditional_entropy_tim})
                logger.add_result(self.run, result)

            torch.save(best_out, f'{self.model_dir}/{self.run}.pt')

        logger.print_statistics()


    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer_inner.zero_grad()
        out = self.model(self.x[self.train_idx])
        loss = self.criterion(out, self.y_true.squeeze(1)[self.train_idx])
        loss.backward()
        self.loss = loss.item()
        self.optimizer_inner.step()
        if epoch >=200:
            outer_grads = self.get_model_grads()
            prob = self.model(self.x[self.split_idx['valid']])
            conditional_entropy = get_cond_entropy(prob)
            self.conditional_entropy_tim = conditional_entropy.item()
            loss2 = nn.functional.tanh(self.args.alpha *conditional_entropy)
            loss2.backward()
            for i,p in enumerate(self.model.lins[-1].parameters()):
                p.data = p.data - self.args.lr * nn.functional.relu(torch.mul(outer_grads[i],p.grad))

            outer_grads = self.get_model_grads()
            prob = self.model(self.x[self.split_idx['valid']])
            kl_divergence = get_kld(self.pi, prob)
            self.kld = kl_divergence.item()
            loss3 = nn.functional.tanh(kl_divergence)
            loss3.backward()
            for i,p in enumerate(self.model.lins[-1].parameters()):
                p.data = p.data - self.args.lr * nn.functional.relu(torch.mul(outer_grads[i],p.grad))
            return loss.item(), loss2.item()
        return loss.item(), 0


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

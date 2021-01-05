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
from utils import to_device, prepare_folder
from loss import get_entropy, get_cond_entropy
from logger import Logger
import wandb

"""
Livia Model
"""
class LTrainer():

    def __init__(self, args):
        self.args = args
        wandb.init(project="TIM", name=f"{args.exp_name}", config=args)
        wandb.config = args
        self.device = self._device()
        self.basic_loss_tim = 0
        self.entropy_tim = 0
        self.conditional_entropy_tim = 0
        self._build()

    def _build(self):
        self._build_loaders()
        self._build_model()
        self._build_criteria_and_optim()
        self._build_scheduler()
        for lmda in np.arange(0.1, 1.1, 0.1):
            for p in np.arange(0.0, 1.1, 0.1):
                self.args.p = p
                self.args.lmda = lmda
                print(f"Using model: {self.args.model}; lambda: {self.args.lmda}; p: {self.args.p}")
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
        elif self.args.criterion == "nll":
            self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.model_dir = prepare_folder(self.args.exp_name, self.model)
        self.evaluator = Evaluator(name=f'ogbn-{self.args.dataset}')

    def _build_scheduler(self):
        rate_decay_step_size = 40
        rate_decay_gamma = 0.8
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=rate_decay_step_size, gamma=rate_decay_gamma)

    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        logger = Logger(self.args.runs, self.args)
        for self.run in range(self.args.runs):
            import gc
            gc.collect()
            # print(sum(p.numel() for p in self.model.parameters()))
            self.model.reset_parameters()
            best_valid = 0
            best_out = None
            for epoch in range(1, self.args.epochs):
                loss = self.train_epoch()
                result, out = self.test_epoch()
                train_acc, valid_acc, test_acc = result
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    best_out = out.cpu().exp()
                if self.args.tim:
                    wandb.log({f"Loss_{self.run}_{self.args.lmda}_{self.args.p}": loss,
                              f"Train_{self.run}_{self.args.lmda}_{self.args.p}": train_acc,
                              f"Valid_{self.run}_{self.args.lmda}_{self.args.p}": valid_acc,
                              f"Test_{self.run}_{self.args.lmda}_{self.args.p}": test_acc,
                              f"Basic_loss_tim_{self.run}_{self.args.lmda}_{self.args.p}": self.basic_loss_tim,
                              f"Entropy_tim_{self.run}_{self.args.lmda}_{self.args.p}": self.entropy_tim,
                              f"Conditional_entropy_tim_{self.run}_{self.args.lmda}_{self.args.p}": self.conditional_entropy_tim})
                else:
                    wandb.log({f"Loss_{self.run}": loss,
                              f"Train_{self.run}": train_acc,
                              f"Valid_{self.run}": valid_acc,
                              f"Test_{self.run}": test_acc})
                logger.add_result(self.run, result)

            # logger.print_statistics(self.run)
            torch.save(best_out, f'{self.model_dir}/{self.run}.pt')

        logger.print_statistics()


    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.x[self.train_idx])
        loss = self.criterion(out, self.y_true.squeeze(1)[self.train_idx])
        if self.args.tim:
            prob = self.model(self.x[self.split_idx['valid']])
            entropy = get_entropy(prob)
            conditional_entropy = get_cond_entropy(prob)
            self.basic_loss_tim = loss.item()
            self.entropy_tim = entropy.item()
            self.conditional_entropy_tim = conditional_entropy.item()
            # loss = loss + (self.args.lmda * (conditional_entropy - (self.args.p * entropy)))
            loss = (self.args.lmda * loss) + (conditional_entropy - (self.args.p * entropy))
        loss.backward()
        self.optimizer.step()

        return loss.item()

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

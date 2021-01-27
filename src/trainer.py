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
from loss import get_entropy, get_cond_entropy, get_kld
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
            self.pi = torch.zeros(40).to(self.device)
            for elem in self.y_true.squeeze(1)[self.split_idx['test']]:
              self.pi[elem]+=1.0
            self.pi = self.pi/len(self.y_true.squeeze(1)[self.split_idx['test']])
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
            self.model.reset_parameters()
            best_valid = 0
            best_out = None
            for epoch in range(1, self.args.epochs):
                loss, loss_tim = self.train_epoch(epoch)
                result, out = self.test_epoch()
                train_acc, valid_acc, test_acc = result
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    best_out = out.cpu().exp()
                # if not self.run:
                #     if self.args.tim:
                #         wandb.log({f"Cross entropy Loss": loss,
                #                   f"Train": train_acc,
                #                   f"Valid": valid_acc,
                #                   f"Test": test_acc,
                #                   f"TIM Loss": loss_tim,
                #                   f"Entropy_tim": self.entropy_tim,
                #                   f"KL_Divergence": self.kld,
                #                   f"Conditional_entropy_tim": self.conditional_entropy_tim})
                #     else:
                #         wandb.log({f"Loss": loss,
                #                   f"Train": train_acc,
                #                   f"Valid": valid_acc,
                #                   f"Test": test_acc})
                logger.add_result(self.run, result)

            torch.save(best_out, f'../configs/{self.args.exp_name}/{self.run}.pt')
            torch.save(self.pi,f'../configs/prior.pt')
            torch.save(self.final_out,f'../configs/final_out,pt')
        logger.print_statistics()


    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.x[self.train_idx])
        loss = self.criterion(out, self.y_true.squeeze(1)[self.train_idx])
        loss.backward()
        self.optimizer.step()
        if self.args.tim and epoch >= 100:
            prob = self.model(self.x[self.split_idx['test']])
            self.final_out = torch.exp(prob).mean(0, keepdim=True)
            entropy = get_entropy(prob)
            kl_divergence = get_kld(self.pi, prob)
            conditional_entropy = get_cond_entropy(prob)
            self.basic_loss_tim = loss.item()
            self.entropy_tim = entropy.item()
            self.conditional_entropy_tim = conditional_entropy.item()
            self.kld = kl_divergence.item()
            loss2 = (self.args.alpha *conditional_entropy) - (self.args.beta * entropy) + (self.args.gamma * kl_divergence)
            loss2.backward()
            for i,p in enumerate(self.model.lins[-1].parameters()):
                p.data = p.data - self.args.lr * p.grad
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

#!/usr/bin/env python3
"""Helpers: seeding, logging, early stopping, and visualization."""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_directory(path):
    os.makedirs(path, exist_ok=True)


def save_args(args, argsfile):
    with open(argsfile, 'w') as f:
        f.writelines('------------------ start ------------------\n')
        for eachArg, value in args.__dict__.items():
            f.writelines(f'{eachArg} : {str(value)}\n')
        f.writelines('------------------- end -------------------')


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def log(self, *args):
        print(*args)
        with open(self.log_file, 'a') as f:
            for arg in args:
                f.write(f'{arg}\r\n')


class EarlyStopping:
    def __init__(self, savepath=None, patience=7, verbose=False, delta=0, log_fn=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False
        self.delta = delta
        self.savepath = savepath
        self.log_fn = log_fn

    def __call__(self, score, model, fold_index):
        if self.best_score == -np.inf:
            self.save_checkpoint(score, model, fold_index)
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.log_fn(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model, fold_index)
            self.best_score = score
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, score, model, fold_index):
        if self.verbose:
            self.log_fn(f'Best checkpoint: ({self.best_score:.6f} --> {score:.6f}). Saving...')
        torch.save(model.state_dict(), os.path.join(self.savepath, f'{model.__class__.__name__}_{fold_index}.pth'))


class EarlyStoppingByLoss:
    def __init__(self, savepath=None, patience=7, verbose=False, delta=0, log_fn=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False
        self.delta = delta
        self.savepath = savepath
        self.log_fn = log_fn

    def __call__(self, score, model, fold_index):
        if score < self.best_score + self.delta:
            self.save_checkpoint(score, model, fold_index)
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            self.log_fn(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, score, model, fold_index):
        if self.verbose:
            self.log_fn(f'Best checkpoint: ({self.best_score:.6f} --> {score:.6f}). Saving...')
        torch.save(model.state_dict(), os.path.join(self.savepath, f'{model.__class__.__name__}_{fold_index}.pth'))


def initialize_weights(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def get_optimizer_params(model):
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p.append(p)
        else:
            weight_p.append(p)
    return weight_p, bias_p


def plot_loss_curve(epochs, train_losses, valid_losses, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o', markersize=3)
    plt.plot(epochs, valid_losses, label='Validation Loss', color='red', marker='x', markersize=3)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.legend(), plt.grid(True, alpha=0.3)
    plt.tight_layout(), plt.savefig(save_path, dpi=150), plt.close()

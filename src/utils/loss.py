#!/usr/bin/env python3
"""Loss functions for classification tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance (Lin et al., 2017)."""
    def __init__(self, num_class, device, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.class_num = num_class
        self.device = device
        self.alpha = Variable(torch.ones(num_class, 1))

    def forward(self, inputs, targets):
        N, C = inputs.size(0), inputs.size(1)
        P = F.softmax(inputs, dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.0)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(self.device)
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -self.alpha[ids.data.view(-1)] * (torch.pow((1 - probs), self.gamma)) * log_p
        return batch_loss.mean()


class PolyLossDrugBAN(nn.Module):
    """PolyLoss with CrossEntropy base (from DrugBAN)."""
    def __init__(self, batch_size, device, epsilon=1.0):
        super().__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=None, reduction='none')
        self.epsilon = epsilon
        self.DEVICE = device

    def forward(self, predicted, labels):
        one_hot = torch.zeros((predicted.shape[0], 2), device=self.DEVICE).scatter_(1, torch.unsqueeze(labels, dim=-1), 1)
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        ce = self.CELoss(predicted, labels)
        return torch.mean(ce + self.epsilon * (1 - pt))


def get_loss_dict(args):
    loss_dict = {}
    if args.num_class == 1:
        loss_dict.update({'BCELoss': nn.BCELoss(), 'MSELoss': nn.MSELoss(), 'L1Loss': nn.L1Loss()})
    elif args.num_class == 2:
        loss_dict.update({
            'CrossEntropyLoss': nn.CrossEntropyLoss(),
            'FocalLoss': FocalLoss(num_class=args.num_class, device=args.device),
            'PolyLoss_drugban': PolyLossDrugBAN(batch_size=args.batch_size, device=args.device),
        })
    elif args.num_class > 2:
        loss_dict.update({'BCEWithLogitsLoss': nn.BCEWithLogitsLoss()})
    return loss_dict

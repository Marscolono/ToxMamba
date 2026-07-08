#!/usr/bin/env python3
"""Collate functions for batching data in DataLoader."""

import torch
import pandas as pd
import numpy as np


class CollateFunc:

    @staticmethod
    def normal_collate_single(batch):
        embeddings = torch.stack([torch.as_tensor(item[0]) for item in batch], dim=0)
        names = pd.concat([pd.Series(item[1]) for item in batch], axis=0)
        labels = torch.stack([torch.tensor(np.array([item[2]], dtype=int)) for item in batch], dim=0).squeeze(1)
        return embeddings, names, labels

    @staticmethod
    def normal_collate_single_regression(batch):
        embeddings = torch.stack([torch.as_tensor(item[0]) for item in batch], dim=0)
        names = pd.concat([pd.Series(item[1]) for item in batch], axis=0)
        labels = torch.stack([torch.tensor(np.array(float(item[2]), dtype=np.float64)) for item in batch], dim=0)
        return embeddings, names, labels

    @staticmethod
    def normal_collate_two(batch):
        input1 = torch.stack([item[0] for item in batch], dim=0)
        input2 = torch.stack([item[1] for item in batch], dim=0)
        names = pd.concat([pd.Series(item[2]) for item in batch], axis=0)
        labels = torch.stack([torch.tensor(item[3]) for item in batch], dim=0)
        return input1, input2, names, labels

    @staticmethod
    def normal_collate_three(batch):
        input1 = torch.stack([item[0] for item in batch], dim=0)
        input2 = torch.stack([item[1] for item in batch], dim=0)
        input3 = torch.stack([item[2] for item in batch], dim=0)
        names = pd.concat([pd.Series(item[3]) for item in batch], axis=0)
        labels = torch.stack([torch.tensor(int(item[4])) for item in batch], dim=0)
        return input1, input2, input3, names, labels

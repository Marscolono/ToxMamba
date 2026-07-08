#!/usr/bin/env python3
"""Dataset classes for loading peptide/protein sequence data."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROT_DICT = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
}

DNA_DICT = {"A": 1, "T": 2, "G": 3, "C": 4}

ADAPT_TOKEN2INDEX = {
    '[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3,
    'B': 4, 'Q': 5, 'I': 6, 'D': 7, 'M': 8, 'V': 9,
    'G': 10, 'K': 11, 'Y': 12, 'P': 13, 'H': 14, 'Z': 15,
    'W': 16, 'U': 17, 'A': 18, 'N': 19, 'F': 20, 'R': 21,
    'S': 22, 'C': 23, 'E': 24, 'L': 25, 'T': 26, 'X': 27
}


def extract_or_pad_middle_sequence(seq, max_len, pad_char='N'):
    seq_length = len(seq)
    if seq_length >= max_len:
        middle_start = (seq_length - max_len) // 2
        return seq[middle_start:middle_start + max_len]
    else:
        pad_left = (max_len - seq_length) // 2
        pad_right = max_len - seq_length - pad_left
        return pad_char * pad_left + seq + pad_char * pad_right


def onehot_encoding(seq, char_dict, max_len):
    one_hot_matrix = np.zeros((max_len, len(char_dict)), dtype=np.float32)
    seq_processed = extract_or_pad_middle_sequence(seq, max_len)
    for i, char in enumerate(seq_processed):
        if char in char_dict:
            one_hot_matrix[i, char_dict[char] - 1] = 1.0
    return torch.from_numpy(one_hot_matrix)


def adapt_encoding(seq, max_len):
    seq_id = [ADAPT_TOKEN2INDEX.get(residue, 0) for residue in seq[:max_len]]
    if len(seq_id) < max_len:
        seq_id.extend([0] * (max_len - len(seq_id)))
    return np.array(seq_id).T


class DatasetsSingle(Dataset):
    """Dataset for single-input peptide/protein sequence data.

    Data format (TSV/CSV with header):
        name<TAB>sequence<TAB>label
    """

    def __init__(self, args, state_=None, index=None, only_pos=False, Test=False):
        self.file_name = args.species
        self.max_len = args.max_len_list[0]
        self.embed_type = args.embed_type

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.seq_path = os.path.join(base_dir, 'data', self.file_name)
        self.esm_path = os.path.join(self.seq_path, 'ESM2_embeddings')

        if args.file_sytle == 'train_test_2file':
            if Test:
                file_path = os.path.join(self.seq_path, 'test.csv')
                if getattr(args, 'independent', False):
                    indep_path = os.path.join(self.seq_path, 'independent_test.csv')
                    if os.path.exists(indep_path):
                        file_path = indep_path
            else:
                file_path = os.path.join(self.seq_path, 'train.csv')
        else:
            file_path = os.path.join(self.seq_path, 'train.csv')

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                first_line = f.readline()
            sep = '\t' if '\t' in first_line else ','
            self.data = pd.read_csv(file_path, sep=sep)
            if self.data.shape[1] < 3:
                self.data = pd.read_csv(file_path, sep=sep, header=None)
        else:
            raise FileNotFoundError(f"Data file not found: {file_path}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        if self.data.shape[1] >= 3:
            name, seq, label = str(row.iloc[0]), str(row.iloc[1]), int(row.iloc[2])
        else:
            name, seq = str(row.iloc[0]), str(row.iloc[1])
            label = 0

        if self.embed_type == 'onehot':
            embedding = onehot_encoding(seq, PROT_DICT, self.max_len)
        elif self.embed_type == 'adapt':
            embedding = torch.from_numpy(adapt_encoding(seq, self.max_len)).long()
        elif self.embed_type in ('pretrain', 'pretrain_ISM'):
            embedding = self._load_pretrain_embedding(name)
        else:
            embedding = torch.from_numpy(adapt_encoding(seq, self.max_len))

        return embedding, name, int(label)

    def _load_pretrain_embedding(self, name):
        pt_path = os.path.join(self.esm_path, f'{name}.pt')
        if os.path.exists(pt_path):
            embedding = torch.load(pt_path, map_location='cpu', weights_only=True)
            if isinstance(embedding, dict):
                embedding = embedding.get('representations', embedding)
                if isinstance(embedding, dict):
                    embedding = embedding.get(33, embedding)
            return embedding.squeeze(0) if embedding.dim() == 3 else embedding
        return torch.zeros(self.max_len, 1280)


class DatasetsSingleGraph(Dataset):
    """Dataset for graph-based protein data."""

    def __init__(self, args, state_=None, index=None, only_pos=False, Test=False):
        self.file_name = args.species
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.seq_path = os.path.join(base_dir, 'data', self.file_name)
        self.graph_path = os.path.join(self.seq_path, 'graphs')
        file_path = os.path.join(self.seq_path, 'test.csv' if Test else 'train.csv')
        if os.path.exists(file_path):
            self.data = pd.read_csv(file_path, sep='\t' if '\t' in open(file_path).readline() else ',', header=None)
        else:
            raise FileNotFoundError(f"Data file not found: {file_path}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        name, seq, label = self.data.iloc[idx, :3]
        tensor_path = os.path.join(self.graph_path, f'{name}.pt')
        graph_data = torch.load(tensor_path, map_location='cpu', weights_only=True)
        return graph_data, name, int(label)

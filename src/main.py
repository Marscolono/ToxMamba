#!/usr/bin/env python3
"""
ToxMamba - Peptide Toxicity Prediction
Main entry point for training and evaluation.
"""

import os
import sys
import time
import shutil
import argparse

import torch
import numpy as np
import random

try:
    import tomllib
except ModuleNotFoundError:
    import toml as tomllib

from src.args import ArgsInput, merge_args_with_input
from src.trainer import Trainer
from src.utils.helpers import seed_everything, save_args, make_directory


def parse_args():
    parser = argparse.ArgumentParser(description='ToxMamba - Peptide Toxicity Prediction')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--tomlpath', required=True, type=str, help='Path to TOML config')
    parser.add_argument('--embed-type', required=True, type=str, help='Embedding type')
    parser.add_argument('--model-name', required=True, type=str, help='Model class name')
    return parser.parse_args()


def main():
    input_args = parse_args()

    with open(input_args.tomlpath, 'r') as f:
        toml_dict = tomllib.load(f)

    args = ArgsInput(toml_dict)
    args = merge_args_with_input(input_args, args)

    torch.set_float32_matmul_precision('medium')
    seed_everything(args.seed)

    print('=' * 60)
    print(f'  ToxMamba - Peptide Toxicity Prediction')
    print(f'  Model: {args.model_name} | Dataset: {args.species}')
    print(f'  Embedding: {args.embed_type} | Seed: {args.seed}')
    print('=' * 60)

    start_time = time.time()
    trainer = Trainer(args)

    if 'train' in args.train_test_mode:
        print('\n' + '=' * 60)
        print('  PHASE 1: Training (5-fold CV)')
        print('=' * 60)
        trainer.train()

        code_dir = os.path.join(args.path, 'code')
        if os.path.exists(code_dir):
            shutil.rmtree(code_dir)
        os.makedirs(code_dir, exist_ok=True)
        shutil.copy(os.path.abspath(__file__), code_dir)
        save_args(args, os.path.join(code_dir, 'config.txt'))

    if 'test' in args.train_test_mode:
        print('\n' + '=' * 60)
        print('  PHASE 2: Independent Test Evaluation')
        print('=' * 60)
        trainer.test()

    elapsed = (time.time() - start_time) / 60
    print(f'\nTotal time: {elapsed:.2f} minutes. Done.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Arguments and configuration management."""

import os
import torch
from src.data.dataset import DatasetsSingle, DatasetsSingleGraph
from src.utils.loss import FocalLoss, PolyLossDrugBAN, get_loss_dict


class ArgsInput:
    """Configuration container merging TOML config with CLI args."""

    def __init__(self, params):
        self._set_attributes(params)
        self._init_derived()

    def _set_attributes(self, params):
        for key, value in params.items():
            if isinstance(value, dict):
                setattr(self, key, ArgsInput(value))
            else:
                setattr(self, key, value)

    def _init_derived(self):
        self.device = torch.device(
            f"cuda:{self.cuda_num}" if torch.cuda.is_available() else "cpu"
        )
        self.out_dir = getattr(self, 'output_dir', './results')

        if not hasattr(self, 'embed_type'):
            self.embed_type = 'adapt'

        self.data_dict = self._get_dataset_config()
        self.input_num, self.task_type = self.data_dict[0], self.data_dict[-1]
        self.max_len_list = self.data_dict[1]
        self.input_style = self.data_dict[2]

        self.screen = f'_screen_{self.indep_name}' if getattr(self, 'independent', False) else ''
        self.K_fold_only = (self.file_sytle in ['train_valid_2file_only5fold', 'train_valid_1file_only5fold'])
        self.drop_last = getattr(self, 'clip_out', False)

        if not hasattr(self, 'early_stop_matric_bi'):
            self.early_stop_matric_bi = 'AUC'
        if getattr(self, 'clip_iner', False):
            self.early_stop_matric_bi = 'Pre'
        if not hasattr(self, 'K_fold'):
            self.K_fold = 5

        self._init_loss_function()
        self._init_data_loader()
        self._init_collate_functions()

        self.test_fold_list = []
        if not hasattr(self, 'teacher_model_name'):
            self.teacher_model_name = 'Demo'
        if not hasattr(self, 'teacher_species'):
            self.teacher_species = self.species

    def _get_dataset_config(self):
        configs = {
            'PTP_SMGCA_data':       ['single', [50], ['Protein'], 'bi_class'],
            'Toxipep_data_0.8':    ['single', [50], ['Protein'], 'bi_class'],
            'Toxipep_data_0.9':    ['single', [50], ['Protein'], 'bi_class'],
            'HyPepTox_Fuse_data':  ['single', [50], ['Protein'], 'bi_class'],
            'ToxTeller_data':       ['single', [50], ['Protein'], 'bi_class'],
            'PeptiTox_data':        ['single', [50], ['Protein'], 'bi_class'],
            'RoxMSRC_data':         ['single', [50], ['Protein'], 'bi_class'],
            'Tox_peptide_Toxpre_2L': ['single', [50], ['Protein'], 'bi_class'],
        }
        return configs.get(self.species, ['single', [50], ['Protein'], 'bi_class'])

    def _init_loss_function(self):
        loss_dict = get_loss_dict(self)
        self.loss_function = loss_dict.get(self.loss_func_name, torch.nn.CrossEntropyLoss())

    def _init_data_loader(self):
        if self.input_num == 'single' and not getattr(self, 'is_graph_model', False):
            self.load_data = DatasetsSingle
        elif self.input_num == 'single' and getattr(self, 'is_graph_model', False):
            self.load_data = DatasetsSingleGraph
        else:
            self.load_data = DatasetsSingle

    def _init_collate_functions(self):
        from src.data.collate import CollateFunc
        self.collate_fn_test = CollateFunc.normal_collate_single
        self.collate_fn_train = self.collate_fn_test


def merge_args_with_input(parser_args, input_args):
    for key, value in vars(parser_args).items():
        setattr(input_args, key, value)
    return input_args

#!/usr/bin/env python3
"""Training, validation, and testing loops for peptide toxicity prediction."""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

from src.utils.helpers import (
    Logger, EarlyStopping, EarlyStoppingByLoss,
    initialize_weights, get_optimizer_params, make_directory, plot_loss_curve
)
from src.utils.metrics import (
    calculate_metrics, calculate_regression_metrics,
    print_metrics_table, print_fold_results,
    save_metrics_to_file, save_predictions, find_best_threshold
)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self._setup_output_path()
        make_directory(os.path.join(self.args.path, 'log_file'))
        make_directory(os.path.join(self.args.path, 'loss_pictures'))
        make_directory(os.path.join(self.args.path, 'label_pred_screen'))

    def _setup_output_path(self):
        self.args.path = os.path.join(
            self.args.out_dir, self.args.species,
            f'{self.args.model_name}_{self.args.loss_func_name}_{self.args.num_class}'
        )

    def _instantiate_model(self):
        from src.models.mamba_classifier import S_Pretrain_emb_Mamba_ISM
        model_cls = globals().get(self.args.model_name, S_Pretrain_emb_Mamba_ISM)
        model = model_cls(self.args).to(self.args.device)
        teacher = model_cls(self.args).to(self.args.device)
        return model, teacher

    def train(self):
        train_val_dataset = self.args.load_data(args=self.args, Test=False)
        kf = KFold(n_splits=self.args.K_fold, shuffle=True, random_state=0)
        index_all = [(i, j) for (i, j) in kf.split(train_val_dataset)]

        for fold_idx in range(self.args.K_fold):
            print(f'\n{"="*60}\n  Training Fold {fold_idx}/{self.args.K_fold}\n{"="*60}')
            train_idx, val_idx = index_all[fold_idx]
            train_fold = Subset(train_val_dataset, train_idx)
            val_fold = Subset(train_val_dataset, val_idx)

            model, teacher_model = self._instantiate_model()
            self.args.model_name = model.__class__.__name__
            self.args.path = os.path.join(
                self.args.out_dir, self.args.species,
                f'{self.args.model_name}_{self.args.loss_func_name}_{self.args.num_class}'
            )
            make_directory(os.path.join(self.args.path, 'log_file'))
            make_directory(os.path.join(self.args.path, 'loss_pictures'))
            make_directory(os.path.join(self.args.path, 'label_pred_screen'))

            self.logger = Logger(os.path.join(self.args.path, 'log_file', f'log_train_valid_{fold_idx}.txt'))
            self.args.log_fn = self.logger.log
            self.logger.log(f'Output Directory: {self.args.path}')
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.log(f'Model: {model}\nParameters: {total_params:,}')

            self._train_fold(model, train_fold, val_fold, fold_idx)

        self.model = model
        return model, teacher_model

    def _train_fold(self, model, train_fold, val_fold, fold_idx):
        args = self.args
        train_loader = DataLoader(train_fold, batch_size=args.batch_size, shuffle=True,
                                  drop_last=args.drop_last, collate_fn=args.collate_fn_train)
        val_loader = DataLoader(val_fold, batch_size=args.batch_size_test, shuffle=False,
                                collate_fn=args.collate_fn_test)
        self.logger.log(f'Train: {len(train_fold)}, Valid: {len(val_fold)}')
        model = model.to(args.device)

        if not getattr(args, 'transfer', False):
            initialize_weights(model)

        weight_p, bias_p = get_optimizer_params(model)
        optimizer = optim.AdamW([{'params': weight_p, 'weight_decay': 1e-4}, {'params': bias_p, 'weight_decay': 0}],
                                lr=args.Learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        loss_function = args.loss_function

        early_stopping_cls = EarlyStoppingByLoss if args.early_stop_matric_bi == 'loss' else EarlyStopping
        early_stopping = early_stopping_cls(savepath=args.path, patience=args.Patience, verbose=True, log_fn=self.logger.log)

        train_losses_epoch, valid_losses_epoch, epoch_list = [], [], []
        for epoch in range(args.epochs):
            epoch_list.append(epoch)
            model.train()
            train_losses = []
            pbar = tqdm(BackgroundGenerator(train_loader), total=len(train_loader), ncols=120)
            for data in pbar:
                pbar.set_description(f"Epoch:{epoch}|Fold:{fold_idx}|<{args.species}|{args.model_name}>")
                data_input, name, label = data[:-2], data[-2], data[-1]
                data_input = [i.to(args.device) for i in data_input]
                label = label.to(args.device)
                output = model.train_model(data_input)
                if args.num_class == 2 and args.task_type == 'bi_class':
                    output = F.softmax(output, dim=1)
                    loss = loss_function(output, label)
                else:
                    loss = loss_function(output.squeeze(), label.float())
                pbar.set_postfix(loss=loss.item())
                train_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_losses_epoch.append(loss.cpu().detach().numpy())
            self.logger.log(f"Epoch:{epoch}|Fold:{fold_idx} | train_loss: {np.average(train_losses):.4f}")

            model.eval()
            with torch.no_grad():
                result_valid, name_list, preds, true_labels, loss_avg = self._evaluate(model, val_loader, fold_idx, 'valid')
                self.logger.log(f'Valid: {str({k: v for k, v in result_valid.items()})}')
            valid_losses_epoch.append(loss_avg.item())

            if args.task_type == 'bi_class':
                scheduler.step(loss_avg)
                score = roc_auc_score(true_labels, preds) if args.early_stop_matric_bi == 'AUC' else loss_avg.item()
            else:
                score = loss_avg.item()

            if early_stopping(score, model, fold_idx):
                self.logger.log('Early stopping triggered.')
                break

        plot_loss_curve(epoch_list, train_losses_epoch, valid_losses_epoch,
                        os.path.join(args.path, 'loss_pictures', f'loss_curve_fold_{fold_idx}.png'))

    def _evaluate(self, model, data_loader, fold_idx, state='valid'):
        loss_function = self.args.loss_function
        name_list, loss_list, predict_values, true_labels = [], [], [], []
        model.eval()
        with torch.no_grad():
            for data in tqdm(BackgroundGenerator(data_loader), total=len(data_loader), ncols=110):
                data_input, name, label = data[:-2], data[-2], data[-1]
                data_input = [i.to(self.args.device) for i in data_input]
                label = label.to(self.args.device)
                output = model.train_model(data_input)
                true_labels.extend(label.cpu().data.numpy())
                name_list.extend(name)
                if self.args.num_class == 2 and self.args.task_type == 'bi_class':
                    loss_list.append(loss_function(output, label.long()).item())
                    output = F.softmax(output, dim=1)[:, 1]
                    predict_values.extend(output.cpu().data.numpy())
                elif self.args.num_class == 1:
                    output = torch.sigmoid(output)
                    loss_list.append(loss_function(output.squeeze(), label.float()).item())
                    predict_values.extend([i[0] for i in output.cpu().data.numpy()])

        loss_avg = torch.mean(torch.Tensor(loss_list))
        make_directory(os.path.join(self.args.path, 'label_pred_screen'))
        save_predictions(name_list, true_labels, predict_values, fold_idx, state,
                         os.path.join(self.args.path, 'label_pred_screen'), screen=self.args.screen)

        if self.args.task_type == 'bi_class':
            if getattr(self.args, 'best_threshold', False) and state == 'test':
                self.args.threshold = find_best_threshold(true_labels, predict_values)
            result = calculate_metrics(predict_values, true_labels, threshold=self.args.threshold)
        else:
            result = calculate_regression_metrics(predict_values, true_labels)

        if 'test' in state:
            print_metrics_table(result, f'{state}_{fold_idx}', self.args.model_name, log_fn=self.args.log_fn)
        save_metrics_to_file(result, os.path.join(self.args.path, 'metrics'), f'{state}_{fold_idx}', screen=self.args.screen)
        return result, name_list, predict_values, true_labels, loss_avg

    def test(self):
        args = self.args
        result_df = pd.DataFrame()
        test_logger = Logger(os.path.join(args.path, 'log_file', 'log_test_ensemble.txt'))
        args.log_fn = test_logger.log

        for fold_idx in range(args.K_fold):
            test_dataset = args.load_data(args=args, Test=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False,
                                     collate_fn=args.collate_fn_test)
            print(f'\n{"="*60}\n  Testing Fold {fold_idx}/{args.K_fold}\n{"="*60}')

            model, _ = self._instantiate_model()
            checkpoint = os.path.join(args.path, f'{args.model_name}_{fold_idx}.pth')
            model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True), strict=False)
            model = model.to(args.device).eval()

            with torch.no_grad():
                result_i, name_list, preds_i, true_labels, _ = self._evaluate(model, test_loader, fold_idx, 'test')
                result_df = pd.concat([result_df, pd.DataFrame(result_i, index=[0])], axis=0).reset_index(drop=True)

        result_df.loc['mean'] = result_df.mean()
        result_path = os.path.join(args.path, f'model_metrics_{args.K_fold}fold{args.screen}.csv')
        result_df.to_csv(result_path)
        args.log_fn(f'\nResults saved to: {result_path}')
        print_fold_results(result_df, args.K_fold, log_fn=args.log_fn)

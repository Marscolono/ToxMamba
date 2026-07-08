#!/usr/bin/env python3
"""Evaluation metrics for classification and regression tasks."""

import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef, roc_curve,
    roc_auc_score, balanced_accuracy_score,
    mean_squared_error, mean_absolute_error
)
import prettytable as pt
import scipy


def calculate_metrics(y_score, y_true, threshold=0.5):
    y_pred = [int(i >= threshold) for i in y_score]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.flatten()
    sen = tp / (fn + tp) if (fn + tp) > 0 else 0
    spe = tn / (fp + tn) if (fp + tn) > 0 else 0
    pre = metrics.precision_score(y_true, y_pred, zero_division=0)
    rec = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc_val = roc_auc_score(y_true, y_score)
    except ValueError:
        auc_val = 0
    try:
        aupr_val = metrics.average_precision_score(y_true, y_score)
    except ValueError:
        aupr_val = 0
    return {
        'sn': round(sen * 100, 3), 'sp': round(spe * 100, 3),
        'acc': round(acc * 100, 3), 'bacc': round(bacc * 100, 3),
        'recall': round(sen * 100, 3), 'precision': round(pre * 100, 3),
        'MCC': round(mcc, 4), 'AUC': round(auc_val, 4),
        'F1': round(f1, 4), 'AUPR': round(aupr_val, 4),
    }


def calculate_regression_metrics(y_pred, y_true):
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mse = mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    try:
        mape = np.mean(np.abs((y_pred - y_true) / y_true))
    except ZeroDivisionError:
        mape = np.inf
    r_val = scipy.stats.pearsonr(y_true, y_pred)[0]
    p_val = scipy.stats.ttest_ind(y_true, y_pred)[1]
    return {
        'MSE': round(mse, 6), 'RMSE': round(np.sqrt(mse), 6),
        'MAE': round(mean_absolute_error(y_true, y_pred), 6),
        'MAPE': mape if np.isfinite(mape) else 'inf',
        'R2': round(r2, 6), 'R': round(r_val, 6), 'P_value': round(p_val, 6),
    }


def find_best_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    gmeans = np.sqrt(tpr * (1 - fpr))
    return thresholds[np.argmax(gmeans)]


def print_metrics_table(metrics_dict, dataset_name, model_name, log_fn=print):
    try:
        tb = pt.PrettyTable()
        tb.field_names = ["Database"] + ["Model"] + list(metrics_dict.keys())
        tb.add_row([dataset_name] + [model_name] + list(metrics_dict.values()))
        log_fn(tb)
    except NameError:
        log_fn(metrics_dict)


def print_fold_results(dataframe, n_folds, log_fn=print):
    all_index = [i for i in range(n_folds)]
    tb = pt.PrettyTable()
    tb.field_names = ["Metric"] + all_index + ["Mean"] + ["Metric"]
    for col in dataframe.columns:
        values = dataframe[col].values
        rounded = [round(v, 4) if not isinstance(v, np.ma.core.MaskedConstant) else np.nan for v in values]
        tb.add_row([col] + rounded + [col])
    log_fn(tb)


def save_metrics_to_file(metrics_dict, outpath, name, screen=''):
    os.makedirs(outpath, exist_ok=True)
    pd.DataFrame(metrics_dict, index=[0]).T.to_csv(
        os.path.join(outpath, f'metrics_{name}{screen}.csv'), header=False, index=True, sep='\t')


def save_predictions(names, true_labels, predictions, fold, state, outpath, screen=''):
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath, f'{state}_result_fold_{fold}{screen}.txt'), 'w') as f:
        for i in range(len(true_labels)):
            f.write(f'{names[i]}\t{true_labels[i]}\t{predictions[i]}\n')

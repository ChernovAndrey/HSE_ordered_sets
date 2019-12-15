import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datasets import get_dataset_loan_credits, get_dataset0, binarization_dataset, norm_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from aggregate_functions import *

prep_feat = ['ApplicantIncome', 'LoanAmount']  # continuous features

use_pattern_structure = False
cv = 10  # for cross validatipnd
if use_pattern_structure:
    ag_func = simple_aggregate_function_ps
else:
    ag_func = simple_aggregate_function


def get_predict(X_train, x_target, y_train, ag_func, index_prep_feat, flag_print_stat=False):
    count_train_plus = len(y_train[y_train == 1])
    count_train = len(X_train)
    count_train_neg = count_train - count_train_plus
    all_intersections = {}
    for j in range(X_train.shape[0]):
        tr_x = X_train[j]
        ind = np.where((tr_x == x_target) & (tr_x == 1))[0]  # inter section features
        if flag_print_stat:
            print('ind:', ind)
        if all_intersections.get(tuple(ind)) is not None:
            if flag_print_stat:
                print('this features processed before')
            continue  # данное пересечение уже рассматривалось
        count_pl = 0
        count_neg = 0
        if use_pattern_structure:  # support for continuous features
            count_pl_cont = 0
            count_neg_cont = 0
        for k in range(X_train.shape[0]):
            if (X_train[k][ind] == x_target[ind]).all():
                if y_train[k] == 0:
                    count_neg += 1
                else:
                    count_pl += 1
            if use_pattern_structure:
                cs = 0.0
                cor = np.exp(
                    -np.sum(np.abs(X_train[k][index_prep_feat] - x_target[index_prep_feat])))  # continuous support
                if not np.isnan(cor):
                    cs += cor
                if y_train[k] == 0:
                    count_neg += cs
                else:
                    count_pl += cs

        support_plus = count_pl / count_train_plus
        support_neg = count_neg / count_train_neg
        if use_pattern_structure:
            support_plus_cont = count_pl / count_train_plus
            support_neg_cont = count_neg / count_train_neg
            all_intersections[tuple(ind)] = [support_plus, support_neg, support_plus_cont, support_neg_cont]
        else:
            all_intersections[tuple(ind)] = [support_plus, support_neg]

    if flag_print_stat:
        print(all_intersections)
    return ag_func(all_intersections)


def naive_approach(X_train, X_val, y_train, index_prep_feat=None):
    y_pred = np.zeros(len(y_val))
    for i in range(X_val.shape[0]):
        y_pred[i], _, _ = get_predict(X_train, X_val[i], y_train, ag_func,
                                      index_prep_feat=index_prep_feat)

    return y_pred


def approach_learn_param(X_train, X_val, y_train, index_prep_feat):
    support_plus_train = np.zeros(len(y_train))
    support_plus_train_cf = np.zeros(len(y_train))
    support_minus_train = np.zeros(len(y_train))
    support_minus_train_cf = np.zeros(len(y_train))

    for i in range(X_train.shape[0]):
        if use_pattern_structure:
            _, support_plus_train[i], support_minus_train[i], support_plus_train_cf[i], support_minus_train_cf[
                i] = get_predict(
                np.delete(X_train, i, axis=0),
                X_train[i],
                np.delete(y_train, i, axis=0),
                ag_func, index_prep_feat=index_prep_feat)
        else:
            _, support_plus_train[i], support_minus_train[i] = get_predict(np.delete(X_train, i, axis=0),
                                                                           X_train[i],
                                                                           np.delete(y_train, i, axis=0),
                                                                           ag_func,
                                                                           index_prep_feat=index_prep_feat)
    # X_log_reg = np.stack([support_plus_train, support_minus_train], axis=1)
    if use_pattern_structure:
        X_log_reg = np.stack([support_plus_train - support_minus_train, support_plus_train_cf - support_minus_train_cf],
                             axis=1)
        clf = LogisticRegression(solver='lbfgs').fit(X_log_reg, y_train)
    else:
        X_log_reg = support_plus_train - support_minus_train
        clf = LogisticRegression(solver='lbfgs').fit(X_log_reg.reshape(-1, 1), y_train)

    support_plus = np.zeros(len(y_val))
    support_minus = np.zeros(len(y_val))
    support_plus_cf = np.zeros(len(y_val))
    support_minus_cf = np.zeros(len(y_val))
    for i in range(X_val.shape[0]):
        if not use_pattern_structure:
            _, support_plus[i], support_minus[i] = get_predict(X_train, X_val[i], y_train, ag_func,
                                                               index_prep_feat=index_prep_feat)
        else:
            _, support_plus[i], support_minus[i], support_plus_cf[i], support_minus_cf[i] = get_predict(X_train,
                                                                                                        X_val[i],
                                                                                                        y_train,
                                                                                                        ag_func,
                                                                                                        index_prep_feat=index_prep_feat)
    # X_test_log_reg = np.stack([support_plus, support_minus], axis=1)
    if not use_pattern_structure:
        X_test_log_reg = support_plus - support_minus
        return clf.predict(X_test_log_reg.reshape(-1, 1))
    X_test_log_reg = np.stack([support_plus - support_minus, support_plus_cf - support_minus_cf], axis=1)
    return clf.predict(X_test_log_reg)


def get_score(y_pred, y_val):
    acc = len(np.where(y_pred.astype(int) == y_val.astype(int))[0]) / len(X_val)
    p, r, f, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
    print('accuracy:', acc)
    print('f1 score:', f)
    return acc, p, r, f, confusion_matrix(y_val, y_pred)


def log_regression(X, y):
    clf = LogisticRegression(solver='lbfgs')
    print(np.mean(cross_val_score(clf, X.fillna(0.0), y, cv=10, scoring='f1')))


if __name__ == "__main__":
    # X_train, X_val, y_train = get_dataset0()
    X_train, X_test, y_train = get_dataset_loan_credits()
    # log_regression(X_train, y_train)

    if use_pattern_structure:
        X_train = norm_dataset(X_train, prep_feat)
    else:
        X_train = binarization_dataset(X_train, prep_feat)

    print('count positive in train:', len(y_train.loc[y_train == 1]))
    print('count negative in train:', len(y_train.loc[y_train == 0]))

    # estimator = naive_approach
    estimator = approach_learn_param

    col = list(X_train.columns)
    index_prep_feat = []
    if use_pattern_structure:
        for feat in prep_feat:
            index_prep_feat.append(col.index(feat))
    X_train = X_train.values
    y_train = y_train.values
    f1 = np.zeros(cv)
    acc = np.zeros(cv)
    precision = np.zeros(cv)
    recall = np.zeros(cv)
    m_conf = np.zeros((cv, 2, 2))
    for i in range(cv):
        print('start cv number: ', i + 1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train)
        y_pred = estimator(X_train, X_val, y_train, index_prep_feat)
        acc[i], precision[i], recall[i], f1[i], m_conf[i] = get_score(y_pred, y_val)

    print('result cross val:')
    print('mean f1:', np.mean(f1))
    print('mean acc:', np.mean(acc))
    print('mean precision:', np.mean(precision))
    print('mean recall:', np.mean(recall))
    print('mean confusion matrix:')
    print(np.mean(m_conf, axis=0))

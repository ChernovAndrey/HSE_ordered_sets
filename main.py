import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datasets import get_dataset_loan_credits, get_dataset0
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

prep_feat = ['ApplicantIncome', 'LoanAmount']


# prep_feat = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
def binarization_dataset(X, prep_feat):
    def bound_split_log1p(x, quant):  # TODO: check nans
        if np.log1p(x) < quant:
            return 1
        else:
            return 0

    # print('information about features:')
    # print(X[prep_feat].describe())
    dstat = {}
    for feat in prep_feat:
        s = np.log1p(X[feat].dropna())
        # print('features: ', feat)
        # print(s.describe())
        # sns.distplot(s)
        # plt.show()
        stat = s.describe()
        dstat[feat] = [stat['25%'], stat['50%'], stat['75%']]

    X.loc[X.loc[:, 'CoapplicantIncome'] < 3.0, 'CoapplicantIncome'] = 0
    X.loc[X.loc[:, 'CoapplicantIncome'] >= 3.0, 'CoapplicantIncome'] = 1

    X.loc[X.loc[:, 'Loan_Amount_Term'] == 360, 'Loan_Amount_Term'] = 1
    X.loc[X.loc[:, 'Loan_Amount_Term'] < 360, 'Loan_Amount_Term'] = 0

    X['LoanAmount_25'] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][0]))
    X['LoanAmount_50'] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][1]))
    X['LoanAmount_75'] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][2]))

    X['ApplicantIncome_25'] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][0]))
    X['ApplicantIncome_50'] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][1]))
    X['ApplicantIncome_75'] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][2]))
    return X.drop(['ApplicantIncome', 'LoanAmount'], axis=1)


def simple_aggregate_function(dict_inter):
    sum_plus = 0
    sum_neg = 0
    for key, value in dict_inter.items():
        sum_plus += value[0]
        sum_neg += value[1]
    return int(sum_plus >= sum_neg), sum_plus, sum_neg
    # if sum_plus >= sum_neg:
    #     return 1, sum_plus - sum_neg
    # return 0, sum_plus - sum_neg


def simple_weight_aggregate_function(dict_inter):
    sum_plus = 0
    sum_neg = 0
    for key, value in dict_inter.items():
        sum_plus += len(key) * value[0]
        sum_neg += len(key) * value[1]
    return int(sum_plus > sum_neg), sum_plus, sum_neg
    # if sum_plus >= sum_neg:
    #     return 1
    # return 0


# как неучитывать одни и те же пересечения n раз
def get_predict(X_train, x_target, y_train, ag_func, flag_print_stat=False):
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
        for k in range(X_train.shape[0]):
            if (X_train[k][ind] == x_target[ind]).all():
                if y_train[k] == 0:
                    count_neg += 1
                else:
                    count_pl += 1

        support_plus = count_pl / count_train_plus
        support_neg = count_neg / count_train_neg

        all_intersections[tuple(ind)] = [support_plus, support_neg]
    if flag_print_stat:
        print(all_intersections)
    return ag_func(all_intersections)


# X_train, X_val, y_train = get_dataset0()

X_train, X_test, y_train = get_dataset_loan_credits()
X_train = binarization_dataset(X_train, prep_feat)
# print(X_train.shape)
# print(y_train.shape)
print('count positive in train:', len(y_train.loc[y_train == 1]))
print('count negative in train:', len(y_train.loc[y_train == 0]))
X_train, X_val, y_train, y_val = train_test_split(X_train.values, y_train.values, test_size=0.15)#, random_state=42)

# print('X train shape:', X_train.shape)
# print('X val shape:', X_val.shape)
# print('y train shape:', y_train.shape)

count_success = 0

support_plus_train = np.zeros(len(y_train))
support_minus_train = np.zeros(len(y_train))
for i in range(X_train.shape[0]):
    _, support_plus_train[i], support_minus_train[i] = get_predict(X_train, X_train[i], y_train,
                                                                   simple_aggregate_function, False)

X_log_reg = np.stack([support_plus_train, support_minus_train], axis=1)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='lbfgs').fit(X_log_reg, y_train)

support_plus = np.zeros(len(y_val))
support_minus = np.zeros(len(y_val))
# y_pred = np.zeros(len(y_val))
for i in range(X_val.shape[0]):
    _, support_plus[i], support_minus[i] = get_predict(X_train, X_val[i], y_train, simple_aggregate_function, False)

X_test_log_reg = np.stack([support_plus, support_minus], axis=1)
# print('kekke ', X_test_log_reg.shape, X_log_reg.shape)
y_pred = clf.predict(X_test_log_reg)
# print(y_pred[:10])
print(y_pred.shape)

# y_pred[y_pred >= 0.5] = 1
# y_pred[y_pred < 0.5] = 0

print(y_val[:100])
print(y_pred[:100])
print('accuracy:', len(np.where(y_pred.astype(int) == y_val.astype(int))[0]) / len(X_val))
print('f1 score:', f1_score(y_val, y_pred))
print('confusion matrix:')
print(confusion_matrix(y_val, y_pred))

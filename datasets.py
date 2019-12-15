import os
import numpy as np
import pandas as pd


def get_dataset0():
    X_train = np.array([[1, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [0, 1, 1, 0, 0]])
    y_train = np.array([1, 1, 1, 0, 0])
    X_val = np.array([1, 1, 0, 0, 1]).reshape(1, -1)
    return X_train, X_val, y_train


def get_dataset_loan_credits():
    path = '/Users/chernovandrey/Desktop/hse/ordered sets/HW_project_FCA/'
    name_y = 'Loan_Status'
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    test = pd.read_csv(os.path.join(path, 'test.csv'))
    y_train = train[name_y].copy()
    y_train.loc[y_train == 'Y'] = 1
    y_train.loc[y_train == 'N'] = 0

    train.loc[train.loc[:, 'Gender'] == 'Male', 'Gender'] = 1
    train.loc[train.loc[:, 'Gender'] == 'Female', 'Gender'] = 0

    train.loc[train.loc[:, 'Married'] == 'Yes', 'Married'] = 1
    train.loc[train.loc[:, 'Married'] == 'No', 'Married'] = 0

    # train.loc[train.loc[:, 'Dependents'] == '3+', 'Dependents'] = 3

    train.loc[train.loc[:, 'Education'] == 'Graduate', 'Education'] = 1
    train.loc[train.loc[:, 'Education'] == 'Not Graduate', 'Education'] = 0

    train.loc[train.loc[:, 'Self_Employed'] == 'Yes', 'Self_Employed'] = 1
    train.loc[train.loc[:, 'Self_Employed'] == 'No', 'Self_Employed'] = 0

    train = pd.get_dummies(train, columns=['Property_Area', 'Dependents'])
    print('train columns:', train.columns)
    # train[train.loc['Gender'] == ]
    train.drop([name_y, 'Loan_ID'], axis=1, inplace=True)
    return train, test, y_train


def binarization_dataset(X, prep_feat):
    # quant = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quant = [0.25, 0.50, 0.75]

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

        dstat[feat] = np.quantile(s, quant)
        # dstat[feat] = [stat['25%'], stat['50%'], stat['75%']]

    X.loc[X.loc[:, 'CoapplicantIncome'] < 3.0, 'CoapplicantIncome'] = 0
    X.loc[X.loc[:, 'CoapplicantIncome'] >= 3.0, 'CoapplicantIncome'] = 1

    X.loc[X.loc[:, 'Loan_Amount_Term'] == 360, 'Loan_Amount_Term'] = 1
    X.loc[X.loc[:, 'Loan_Amount_Term'] < 360, 'Loan_Amount_Term'] = 0
    for j in range(len(quant)):
        X['LoanAmount_' + str(quant[j])] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][j]))

    # X['LoanAmount_25'] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][0]))
    # X['LoanAmount_50'] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][1]))
    # X['LoanAmount_75'] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][2]))
    for j in range(len(quant)):
        X['ApplicantIncome_' + str(quant[j])] = X['ApplicantIncome'].apply(
            lambda x: bound_split_log1p(x, dstat[feat][j]))

    # X['ApplicantIncome_25'] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][0]))
    # X['ApplicantIncome_50'] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][1]))
    # X['ApplicantIncome_75'] = X['LoanAmount'].apply(lambda x: bound_split_log1p(x, dstat[feat][2]))
    return X.drop(['ApplicantIncome', 'LoanAmount'], axis=1)


def norm_dataset(X, prep_feat):
    for feat in prep_feat:
        mu = np.mean(X[feat].dropna())
        sigma = np.sqrt(np.var(X[feat].dropna(), ddof=1))
        X[feat] = (X - mu) / sigma
    return X

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

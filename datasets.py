import os
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

def preprocess_adult(file_name_train, file_name_test):
    data = arff.loadarff(file_name_train)
    data_test = arff.loadarff(file_name_test)
    df = pd.DataFrame(data[0])
    df_test = pd.DataFrame(data_test[0])
    df['split'] = 'Train'
    df_test['split'] = 'Test'
    df = pd.concat([df, df_test], ignore_index=True)

    numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                   'native-country']
    binary_vbles = ['sex', 'class']
    df[categorical] = df[categorical].applymap(lambda x: x.decode('utf-8'))

    df["workclass"] = df["workclass"].mask(df["workclass"] == '?', np.NaN)
    df["native-country"] = df["native-country"].mask(df["native-country"] == '?', np.NaN)
    df["occupation"] = df["occupation"].mask(df["occupation"] == '?', np.NaN)

    df.dropna(inplace=True)

    # One Hot Encoder
    for cat_vble in categorical:
        one_hot_encoded = pd.get_dummies(df[cat_vble])
        df = pd.concat([one_hot_encoded, df], axis=1, sort=False)
        df = df.drop(cat_vble, axis = 1)

    # Dummy variables
    for binary_vble in binary_vbles:
        df[binary_vble] = LabelEncoder().fit_transform(df[binary_vble])
        df[binary_vble] = df[binary_vble].astype(int)

    # Numerical variables
    for c in numerical:
        df[c] = MinMaxScaler().fit_transform(df[c].values.reshape(-1, 1))

    df['class'] = df['class'].replace({'>50K': 0, '<=50K': 1})

    X_train = df.loc[df['split'] == 'Train']
    X_test = df.loc[df['split'] == 'Test']
    Y_train = X_train['class']
    Y_test = X_test['class']
    X_train = X_train.drop(['class', 'split'], axis = 1)
    X_test = X_test.drop(['class', 'split'], axis = 1)
    binary_vbles_mask = [1 if X_test[c].unique().shape[0] == 2 else 0 for c in X_test.columns]
    return X_train, X_test, Y_train, Y_test, binary_vbles_mask

def preprocess_hypothyroid(file_name_train, file_name_test):
    data = arff.loadarff(file_name_train)
    data_test = arff.loadarff(file_name_test)
    df = pd.DataFrame(data[0])
    df_test = pd.DataFrame(data_test[0])
    df['split'] = 'Train'
    df_test['split'] = 'Test'
    df = pd.concat([df, df_test], ignore_index=True)

    df = df.applymap(lambda x: x.decode() if hasattr(x, 'decode') else x)
    # drop column with missing values

    df = df.drop('TBG', axis=1)
    df = df.drop('TBG_measured', axis=1) # column with just one value
    df = df.replace('?', np.nan) # convert ? to nan

    # fill columns with missing values with the nearest neighbour
    missing_values_columns = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    #for c_missing in missing_values_columns:
        #df[c_missing] = df[c_missing].fillna(df[c_missing].median())
    imputer = KNNImputer(n_neighbors=1)
    df[missing_values_columns] = imputer.fit_transform(df[missing_values_columns].values)

    df['sex'] = df['sex'].fillna(df['sex'].mode().values[0])

    ### dummy variables
    binary_vbles = ['sex', 'on_thyroxine', 'query_on_thyroxine',
                     'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
                     'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                     'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured',
                     'T3_measured', 'TT4_measured', 'T4U_measured',
                     'FTI_measured']
    for c in binary_vbles:
        df[c] = LabelEncoder().fit_transform(df[c])
        df[c] = df[c].astype(int)

    one_hot_referral = pd.get_dummies(df['referral_source'])
    df = pd.concat([one_hot_referral, df], axis=1, sort=False)
    df = df.drop('referral_source', axis = 1)

    numeric_vbles = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    for c in numeric_vbles:
        df[c] = MinMaxScaler().fit_transform(df[c].values.reshape(-1, 1))

    df['Class'] = df['Class'].replace({'negative': 0,
                                       'compensated_hypothyroid': 1,
                                       'primary_hypothyroid': 2,
                                       'secondary_hypothyroid': 3})

    X_train = df.loc[df['split'] == 'Train']
    X_test = df.loc[df['split'] == 'Test']
    Y_train = X_train['Class']
    Y_test = X_test['Class']
    X_train = X_train.drop(['Class', 'split'], axis = 1)
    X_test = X_test.drop(['Class', 'split'], axis = 1)
    binary_vbles_mask = [ 1 if X_test[c].unique().shape[0] == 2 else 0 for c in X_test.columns]
    return X_train, X_test, Y_train, Y_test, binary_vbles_mask

def preprocess_pen_based(file_name_train, file_name_test):

    data = arff.loadarff(file_name_train)
    data_test = arff.loadarff(file_name_test)
    df = pd.DataFrame(data[0])
    df_test = pd.DataFrame(data_test[0])
    df['split'] = 'Train'
    df_test['split'] = 'Test'
    df = pd.concat([df, df_test], ignore_index=True)

    df = df.applymap(lambda x: x.decode() if hasattr(x, 'decode') else x)

    numeric_vbles = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9',
                     'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16']
    for c in numeric_vbles:
        df[c] = MinMaxScaler().fit_transform(df[c].values.reshape(-1, 1))

    X_train = df.loc[df['split'] == 'Train']
    X_test = df.loc[df['split'] == 'Test']
    Y_train = X_train['a17']
    Y_test = X_test['a17']
    X_train = X_train.drop(['a17', 'split'], axis = 1)
    X_test = X_test.drop(['a17', 'split'], axis = 1)
    binary_vbles_mask = [1 if X_test[c].unique().shape[0] == 2 else 0 for c in X_test.columns]
    return X_train, X_test, Y_train.astype(int), Y_test.astype(int), binary_vbles_mask
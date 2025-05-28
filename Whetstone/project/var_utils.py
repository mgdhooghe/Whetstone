import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


def train_on_data(data):
    reg = RandomForestClassifier(n_estimators=500)
    #reg = MLPClassifier()
    #reg = LogisticRegression()
    
    x_train = data[data.columns[:-1]].astype(float)
    y_train = data[data.columns[-1]].astype(float)
    
    if 'fnlwgt' in data.columns:
            weight = x_train['fnlwgt']
            x_train_with_weight = x_train.copy(deep=True)
            x_train = x_train.drop(columns=['fnlwgt'])
    else:
            weight = None
    
    try:
           #reg.fit(x_train,y_train, sample_weight = weight)
           reg.fit(x_train,y_train)
    except Exception as e:
            print("FAILED TRAINING: ",e)
            reg = -1
    
    if 'fnlwgt' in data.columns:
            x_train = x_train_with_weight
    
    return reg
    
def predict_output(data, reg):
    x = data[data.columns[:-1]]
    x = pd.DataFrame(x).reset_index(drop=True)
    if 'fnlwgt' in x.columns:
        x_with_weight = x.copy(deep=True)
        x = x.drop(columns = ['fnlwgt'])
    if reg == -1:
            y_pred = np.zeros([len(data)])
    else:
            try:
                    y_pred = reg.predict(x)
            except:
                y_pred = np.zeros([len(data)])
    if 'fnlwgt' in data.columns:
        x = x_with_weight

    return pd.concat([x,pd.Series(y_pred,name=data.columns[-1])], axis=1)

def get_labels(input_file, protected_var):
    #print(input_file)
    #print(protected_var)
    if 'adult' in input_file:
        if protected_var=='race':
            protected = 'race'
            privileged = 'White'
            unprivileged = 'Other'
        if protected_var=='gender':
            protected = 'sex'
            privileged = 'Male'
            unprivileged = 'Female'
        predicted = 'income'
        preferred = '>50K'
        unpreferred = '<=50K'
    elif 'preproc_compas' in input_file or 'PROPUBLICA' in input_file or 'propublica' in input_file or 'compas' in input_file:
        if protected_var=='race':
            protected='race'
            privileged='Caucasian'
            unprivileged = 'Other'
            #privileged='1.0'
            #unprivileged='0.0'
        if protected_var=='gender':
            protected='sex'
            privileged='Male'
            unprivileged = 'Female'
        predicted='two_year_recid'
        preferred='0'
        unpreferred='1'
    elif 'census' in input_file:
        if protected_var=='race':
            protected='RACE'
            privileged='White'
            unprivileged = 'Other'
        if protected_var=='gender':
            protected='SEX'
            privileged='Male'
            unprivileged = 'Female'
        predicted='INCOME_50K'
        preferred='50000+.'
        unpreferred='-50000.'
    elif 'german' in input_file:
        if protected_var=='gender':
            protected='gender'
            privileged='male'
            unprivileged='female'
        if protected_var=='age':
            protected='Age'
            privileged='1'
            unprivileged='0'
        predicted='labels'
        preferred='1'
        unpreferred='0'
    elif 'bank' in input_file:
        print('inbank')
        if protected_var=='marital':
            protected='marital'
            privileged='married'
            unprivileged='single'
        if protected_var=='age':
            protected='age'
            privileged='1'
            unprivileged='0'
        predicted='y'
        preferred='yes'
        unpreferred='no'
    elif 'medical' in input_file:
        if protected_var=='race':
            protected='RACE'
            privileged='1'
            unprivileged='0'
        predicted='UTILIZATION'
        preferred='1'
        unpreferred='0'
    return protected, privileged, unprivileged, predicted, preferred, unpreferred
    
def get_og_file(input_file, protected):
    if 'adult' in input_file:
        og_file = 'real_data/adult/ADULT-SPLIT-TRAIN-60.csv'
        og_test = 'real_data/adult/ADULT-SPLIT-TEST-20.csv'
        VGAN_norm = '../gan_data/adult/adult-VGAN-1hot-norm-split-60/9/'
        VGAN_gmm = '../gan_data/adult/adult-VGAN-1hot-gmm-split-60/8-9/'
        WGAN_norm = '../gan_data/adult/adult-VGAN-WTRAIN-split/'
        WGAN_gmm = '../gan_data/adult/adult-VGAN-1hot-gmm-split-60-WTrain-best/90/' 
        VGAN_ordi_norm = '../gan_data/adult/adult-VGAN-ordi-norm-split-60/8-9/'
        VGAN_ordi_gmm = '../gan_data/adult/adult-VGAN-ordi-gmm-split-60/8-9/'
        TABFAIRGAN = '../gan_data/adult/adult-TABFAIRGAN/'
        FAIRGAN = '../gan_data/adult/adult-FAIRGAN/'
    elif 'compas' in input_file:
        og_test = 'propublica-compas/PROPUBLICA-COMPAS-SPLIT-TEST-20.csv'
        og_file = 'propublica-compas/PROPUBLICA-COMPAS-SPLIT-TRAIN-60.csv'
        VGAN_norm = '../gan_data/compas/compas-pro-VGAN-1hot-norm-split-valid-best/9/' 
        VGAN_gmm = '../gan_data/compas/compas-pro-VGAN-1hot-gmm-split-60/8-9/'
        WGAN_norm = '../gan_data/compas/compas-pro-VGAN-WTRAIN-split/'
        WGAN_gmm = '../gan_data/compas/compas-pro-VGAN-1hot-gmm-split-60-WTrain/8-9/'
        VGAN_ordi_norm = '../gan_data/compas/compas-pro-VGAN-ordi-norm-split-60/8-9/'
        VGAN_ordi_gmm = '../gan_data/compas/compas-pro-VGAN-ordi-gmm-split-60/8-9/'
        TABFAIRGAN = '../gan_data/compas/compas-pro-TABFAIRGAN/' + protected + '/'
        FAIRGAN = '../gan_data/compas/compas-pro-FAIRGAN/' + protected + '/'
    elif 'german' in input_file:
        og_file = 'german/GERMAN-SPLIT-TRAIN-60.csv'
        og_test = 'german/GERMAN-SPLIT-TEST-20.csv'
        VGAN_norm = '../german/german-VGAN-1hot-split-validation-best/9/'
        VGAN_gmm = '../german/german-VGAN-1hot-gmm-split-60/8-9/'
        WGAN_norm = '../german/german-VGAN-WTRAIN-split/'
        WGAN_gmm = '../german/german-VGAN-1hot-gmm-split-60-WTrain/8-9/'
        TABFAIRGAN = '../german/german-TABFAIRGAN/'
        FAIRGAN = '../german/german-FAIRGAN/'
    elif 'bank' in input_file:
        og_file = 'real_data/bank/bank-full-SPLIT-TRAIN-60-age.csv'
        og_test = 'real_data/bank/bank-full-SPLIT-TEST-20-age.csv'
        VGAN_norm = '../gan_data/bank/bank-VGAN-best/9-age/'
        VGAN_gmm = '../gan_data/bank/bank-full-VGAN-1hot-gmm-split-60/8-9/'
        WGAN_norm = '../gan_data/bank/bank-VGAN-WTRAIN-split/'
        WGAN_gmm = '../gan_data/bank/bank-full-VGAN-1hot-gmm-split-60-WTrain/8-9/'
        VGAN_ordi_norm = '../gan_data/bank/bank-full-VGAN-ordi-norm-split-60/8-9/'
        VGAN_ordi_gmm = '../gan_data/bank/bank-full-VGAN-ordi-gmm-split-60/8-9/'
        TABFAIRGAN = '../gan_data/bank/bank-TABFAIRGAN/'
        FAIRGAN = '../gan_data/bank/bank-FAIRGAN/'
    elif 'medical' in input_file:
        og_file = 'real_data/medical/meps21-SPLIT-TRAIN-60.csv'
        og_test = 'real_data/medical/meps21-SPLIT-TEST-20.csv'
        VGAN_norm = '../gan_data/medical/medical-VGAN-1hot-norm-split-60/8-9/'
        VGAN_gmm = '../gan_data/medical/medical-VGAN-1hot-gmm-split-60/8-9/'
        WGAN_norm = '../gan_data/medical/medical-WTRAIN-VGAN-1hot-norm-split-60/8-9/'
        WGAN_gmm = '../gan_data/medical/medical-WTRAIN-VGAN-1hot-gmm-split-60-best/8-9/'
        VGAN_ordi_norm = '../gan_data/medical/medical-VGAN-ordi-norm-split-60/8-9/'
        VGAN_ordi_gmm = '../gan_data/medical/medical-VGAN-ordi-gmm-split-60/8-9/'
        TABFAIRGAN = '../gan_data/medical/medical-TABFAIRGAN/'
        FAIRGAN = '../gan_data/medical/medical-FAIRGAN/'
    return og_file, og_test, VGAN_norm, VGAN_gmm, WGAN_norm, WGAN_gmm, VGAN_ordi_norm, VGAN_ordi_gmm, TABFAIRGAN, FAIRGAN


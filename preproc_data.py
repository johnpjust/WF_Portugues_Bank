import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics

############## one-hot encoding function ###############
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)

def return_data(orig=False):
    with open(r'C:\Users\justjo\Downloads\bank.csv', 'r') as infile:
        data = infile.read()
        data = data.replace('"', "")
        data = data.replace('NULL', "")

    header = data.splitlines()[0].split(';')[:-1]

    df = pd.DataFrame([x.split(';')[:-1] for x in data.splitlines()][1:], columns=header)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    for col in df.columns:
        try:
            df[col] = df[col].astype(np.float)
        except:
            pass

    if orig:
        return df, None, None

    df.age.fillna(0, inplace=True)
    df.replace([''], 'unknown', inplace=True)
    df.replace(['other'], 'unknown', inplace=True)
    df.pdays.replace(-1, 1000, inplace=True)

    # ######### double check ########
    # df = pd.read_csv(r'D:\Personal\presentations\WF\bank_nopar.csv', sep=',')
    # df = df.drop('Validation', axis=1)
    # df = df.drop('reweight', axis=1)
    # ################################
    # df = df.sample(frac=1)
    lb = preprocessing.LabelBinarizer()
    labels = df.y.copy()
    labels = lb.fit_transform(labels)
    df = df.drop('y', axis=1)
    pos_weight = len(labels)/np.sum(labels)

    cols_to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact','month','poutcome']
    for col in cols_to_encode:
        df = encode_and_bind(df, col)

    return df, labels, pos_weight
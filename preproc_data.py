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

    cols_to_encode = ['job', 'marital', 'education', 'default', 'housing',
           'loan', 'contact','month','poutcome']

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

    for col in cols_to_encode:
        df = encode_and_bind(df, col)

    lb = preprocessing.LabelBinarizer()
    labels = df.y.copy()
    labels = lb.fit_transform(labels)
    df = df.drop('y', axis=1)
    pos_weight = len(labels)/np.sum(labels)

    return df.sample(frac=1), labels, pos_weight
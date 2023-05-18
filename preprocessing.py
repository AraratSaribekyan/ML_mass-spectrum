import numpy as np
import pandas as pd
import json
from IPython.display import display

def read_data(f):
    with open(f, 'rb') as fp:
        test_d = json.load(fp)
        test_df = pd.DataFrame(json.loads(test_d)).T
    return test_df


def create_speactr(mz, intens):
    spec = []
    for i in range(200, 1750):
        if i in mz:
            spec.append(intens[mz.index(i)])
        else:
            spec.append(0)
    return spec

def prepocess_data(data):
    data['mz'] = data['m/z'].apply(lambda x: [int(x_i // 10) for x_i in x])
    data['intens'] = data.apply(lambda d: create_speactr(d['mz'], d['Rel. Intens.']),
                           axis = 1)
    return data


def make_x(df, col):
    X = []
    for i in df.index:
        row = df.loc[i, col]
        X.append(row)
    return np.array(X)


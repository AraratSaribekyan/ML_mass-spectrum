import numpy as np
import pandas as pd
import json

with open('train.json', 'rb') as fp:
    train_d = json.load(fp)
train_df = pd.DataFrame(json.loads(train_d)).T
def get_dif(x):
    return max(x) - min(x)

def generate_new(train_df):
    train_df.loc[:, 'n_peak'] = train_df['m/z'].apply(len)
    peaks_dif = train_df.groupby(['strain']).agg({'n_peak': get_dif})
    mean_dif = int(peaks_dif.n_peak.mean())
    FEATURES = list(train_df.keys())
    del FEATURES[0]
    del FEATURES[len(FEATURES)-1]
    np.random.seed(142)
    train_gen = pd.DataFrame()
    for strain in train_df.strain.unique():
        tmp = train_df[train_df.strain == strain]
        s = np.random.randint(max(tmp.n_peak) - mean_dif, max(tmp.n_peak), 9)
        strain_df = pd.DataFrame()
        for i, sample in enumerate(tmp[FEATURES].values):
            tmp_i = pd.DataFrame(list(sample)).T
            tmp_i.columns = FEATURES
            strain_df = pd.concat([strain_df, tmp_i])

        for i in range(0, 9):
            df_i = pd.DataFrame()
            for n in range(0, s[i]):
                if isinstance(strain_df.loc[n], pd.Series):
                    continue
                else:
                    peaks = list(strain_df['Rel. Intens.'].loc[n])
                    if 1 in peaks:
                        if 1 in list(df_i['Rel. Intens.']):
                            continue
                        else:
                            index = peaks.index(1)
                            df_i = pd.concat([df_i, pd.DataFrame(strain_df.loc[n].iloc[index]).T])
                    else:
                        df_i = pd.concat([df_i, strain_df.loc[n].sample(n=1)])
            df_i[['id']] = i
            df_i.loc[:, 'strain'] = tmp.strain.unique()[0]
            df_i.loc[:, 'n_peak'] = n
            train_gen = pd.concat([train_gen, df_i])

    train_gen = train_gen.groupby(['strain', 'id']).agg(list).reset_index().drop(['id'], axis=1)
    train_gen.loc[:, 'n_peak'] = train_gen.n_peak.apply(min)

    new_train_df = pd.concat([train_df, train_gen[train_df.columns]])
    new_train_df = new_train_df.reset_index().drop(['index'], axis=1)
    return new_train_df

train_gen = generate_new(train_df)
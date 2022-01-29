import numpy as np
import pandas as pd

import os
import os.path as osp
import config
from sklearn.model_selection import TimeSeriesSplit


def cross_validation(df):

    tss = TimeSeriesSplit(n_splits=4)

    df.loc[:,'fold'] = 0

    for fold, (train_idx, valid_idx) in enumerate(tss.split(df)):
        df.loc[valid_idx,'fold'] = fold

    df.to_csv(osp.join(config.ROOT_DIR,"data","processed","train_folds.csv"),index=False)
    print('-'*20 + 'Done' +'-'*20)

if __name__ == '__main__':
    df = pd.read_csv(osp.join(config.ROOT_DIR,"data","raw","train.csv"))
    cross_validation(df=df)
import os
import config
import numpy as np
import pandas as pd

from model_dispatcher import models
from utils import SMAPE

import joblib
import time

def train_folds(df, model_name):
    # separate cols
    target_cols = ['num_sold']
    feature_cols = df.drop(columns=['row_id','fold','num_sold']).columns.tolist()
    
    # feature and target
    xall = df[feature_cols]
    # yall = df[target_cols].values.ravel()

    # transform 
    yall = np.log1p(df[target_cols].values.ravel() / df.gdp.values.ravel())

    # model training
    model = models[model_name]
    # model train
    model.fit(xall,yall)
    # evaluate
    y_pred_train =  model.predict(xall)
    
    # transform
    yall = df[target_cols].values.ravel()
    y_pred_train = np.ceil(np.expm1(y_pred_train) * df.gdp.values.ravel())

    # metrics
    smape_train = SMAPE(yall, y_pred_train)

    print(f"model={model_name}, SMAPE: all {smape_train:.4f}")

    # save the model
    joblib.dump(model,os.path.join(config.ROOT_DIR,"models","all",f"{model_name}_all.joblib"))

    return y_pred_train


if __name__ == '__main__':
    start_time = time.time()
    # model_names = ['rf','xgb', 'cat']
    # model_names = ['rf','cat']
    # model_names = ['cat']
    # model_names = ['rf']
    # model_names = ['xgb']
    model_names = ['stack']



    df = pd.read_csv(os.path.join(config.ROOT_DIR,"data","processed","train_feat_eng_01.csv"))
    
    for model_name in model_names:
        train_folds(df=df,model_name=model_name)

    print(f"--- {time.time() - start_time:.4f} seconds ---")
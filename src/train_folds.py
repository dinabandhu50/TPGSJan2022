import os
import config
import numpy as np
import pandas as pd

from model_dispatcher import models
from utils import SMAPE

import joblib
import time


def train_folds(fold, df, model_name):
    # separate cols
    target_cols = ['num_sold']
    feature_cols = df.drop(columns=['row_id','fold','num_sold']).columns.tolist()
    
    # separate train valid
    df_train = df[df.fold != fold]
    df_valid = df[df.fold == fold]
    
    # feature and target
    xtrain = df_train[feature_cols]
    xvalid = df_valid[feature_cols]
    # ytrain = df_train[target_cols].values.ravel()
    # ytrain = np.log1p(df_train[target_cols] / df_train.gdp)
    ytrain = np.log1p(df_train[target_cols].values.ravel() / df_train.gdp.values.ravel())

    # yvalid = df_valid[target_cols].values.ravel()
    # print(np.log1p(df_valid[target_cols].values.ravel() / df_valid.gdp.values.ravel()))
    # yvalid = np.log1p(df_valid[target_cols] / df_valid.gdp)
    yvalid = np.log1p(df_valid[target_cols].values.ravel() / df_valid.gdp.values.ravel())
    # print(yvalid)


    # model training
    model = models[model_name]

    # model train
    if model_name == 'xgb':
        model.fit(xtrain, ytrain, eval_set = [[xtrain, ytrain],[xvalid, yvalid]], verbose = False)
    elif model_name == 'cat':
        model.fit(xtrain,ytrain, eval_set = (xvalid, yvalid), early_stopping_rounds=1000)
    else:
        model.fit(xtrain,ytrain)
    

    # evaluate
    y_pred_train =  model.predict(xtrain)
    y_pred_valid =  model.predict(xvalid)

    # conversion
    ytrain = df_train[target_cols].values.ravel()
    y_pred_train = np.ceil(np.expm1(y_pred_train) * xtrain.gdp.values.ravel())
    yvalid = df_valid[target_cols].values.ravel()
    y_pred_valid = np.ceil(np.expm1(y_pred_valid) * xvalid.gdp.values.ravel())
    
    # metrics
    smape_train = SMAPE(ytrain, y_pred_train)
    smape_valid = SMAPE(yvalid, y_pred_valid)


    print(f"model={model_name}, fold={fold}, SMAPE: train {smape_train:.4f}, valid {smape_valid:.4f}")

    # save the model
    joblib.dump(model,os.path.join(config.ROOT_DIR,"models",f"{model_name}_{fold}.joblib"))

    result_df = pd.concat(
        [
            df_valid[['row_id','fold']].reset_index(drop=True),
            pd.DataFrame(y_pred_valid,columns=target_cols),
            pd.DataFrame(yvalid,columns=target_cols),
        ],
        axis=1,
    )
    fold_dict = {
        "train_smape" : smape_train,
        "valid_smape" : smape_valid
    }
    return result_df, fold_dict


if __name__ == '__main__':
    # model_names = ['rf','xgb', 'cat']
    # model_names = ['rf']
    # model_names = ['rf','cat']
    # model_names = ['cat']
    model_names = ['xgb']

    folds = 4
    df = pd.read_csv(os.path.join(config.ROOT_DIR,"data","processed","train_feat_eng_01.csv"))
    
    for model_name in model_names:
        start_time = time.time()
        smape_train_avg = []
        smape_valid_avg = []

        dfs = []
        for fold in range(folds):
            fold_df, fold_dict = train_folds(fold=fold,df=df,model_name=model_name)
            dfs.append(fold_df)
            smape_train_avg.append(fold_dict['train_smape'])
            smape_valid_avg.append(fold_dict['valid_smape'])

        dfs = pd.concat(dfs)
        print(f'model={model_name}, Average, smape_train={np.mean(smape_train_avg):2.7f} \u00B1 {np.std(smape_train_avg):2.7f}, f1_valid={np.mean(smape_valid_avg):2.7f} \u00B1 {np.std(smape_valid_avg):2.7f} ')
        dfs.to_csv(os.path.join(config.ROOT_DIR, "oofs",f"{model_name}_preds.csv"), index=False)
        print(f"---------- {time.time() - start_time:.4f} seconds ----------")
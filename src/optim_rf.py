import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import config
import logging
import optuna
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from utils import SMAPE


def train_folds(fold, df, param):
    # separate cols
    target_cols = ['num_sold']
    feature_cols = df.drop(columns=['row_id','fold','num_sold']).columns.tolist()
    
    # separate train valid
    df_train = df[df.fold != fold]
    df_valid = df[df.fold == fold]
    
    # feature and target
    xtrain = df_train[feature_cols]
    # ytrain = df_train[target_cols].values.ravel()
    ytrain = np.log1p(df_train[target_cols].values.ravel() / df_train.gdp.values.ravel())

    xvalid = df_valid[feature_cols]
    # yvalid = df_valid[target_cols].values.ravel()
    yvalid = np.log1p(df_valid[target_cols].values.ravel() / df_valid.gdp.values.ravel())


    # model training
    model = RandomForestRegressor(**param)

    # model train
    model.fit(xtrain,ytrain)
        
    # evaluate
    # y_pred_train =  model.predict(xtrain)
    y_pred_valid =  model.predict(xvalid)

    # conversion
    # ytrain = df_train[target_cols].values.ravel()
    # y_pred_train = np.ceil(np.expm1(y_pred_train) * xtrain.gdp.values.ravel())
    yvalid = df_valid[target_cols].values.ravel()
    y_pred_valid = np.ceil(np.expm1(y_pred_valid) * xvalid.gdp.values.ravel())
    
    
    # metrics
    # smape_train = SMAPE(ytrain, y_pred_train)
    smape_valid = SMAPE(yvalid, y_pred_valid)


    # print(f"model=RF, fold={fold}, SMAPE: train {smape_train:.4f}, valid {smape_valid:.4f}")


    fold_dict = {
        # "train_smape" : smape_train,
        "valid_smape" : smape_valid
    }
    return fold_dict


def objective(trial):

    param = {
        'random_state':42,
        # "n_jobs": -1,
        # 'class_weight':'balanced',
        "n_estimators" : trial.suggest_int("n_estimators", 100, 600),
        # "criterion" : trial.suggest_categorical('criterion', ['squared_error','absolute_error','poisson']),
        # "criterion" : trial.suggest_categorical('criterion', ['squared_error','absolute_error']),
        "max_depth" : trial.suggest_int("max_depth", 2, 16),
        "max_features":trial.suggest_categorical("max_features",["sqrt", "log2"]),
        # "ccp_alpha":trial.suggest_float("ccp_alpha", 0.0001, 1),
        "min_samples_split": trial.suggest_float("min_samples_split",0.01,0.8),
        "min_samples_leaf": trial.suggest_float("min_samples_leaf",0.01,0.5),
    }
       
    df = pd.read_csv(os.path.join(config.ROOT_DIR,"data","processed","train_feat_eng_01.csv"))

    folds = 4
    # smape_train_avg = []
    smape_valid_avg = []
    for fold in range(folds):
        fold_dict = train_folds(fold=fold,df=df,param=param)
        # smape_train_avg.append(fold_dict['train_smape'])
        smape_valid_avg.append(fold_dict['valid_smape'])

    return np.mean(smape_valid_avg)


if __name__ == '__main__':

    study = optuna.create_study(
        sampler = optuna.samplers.TPESampler(seed=735),
        direction="minimize"
        )
    study.optimize(objective, n_trials=10, n_jobs= -1)

    print("best trials:")
    trial_ = study.best_trial

    print(trial_.values) 
    print(trial_.params)
    
    logging.basicConfig(
        filename=os.path.join(config.ROOT_DIR,"logs",f"rf_param.log"),
        filemode='a',
        level=logging.INFO)
    logging.info(f'\n values = {trial_.values} \n param = {trial_.params}')


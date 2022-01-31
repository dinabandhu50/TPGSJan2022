import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import config
import numpy as np
import pandas as pd
import logging
import optuna

from xgboost import XGBRegressor
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
    model = XGBRegressor(**param)

    # model train
    model.fit(xtrain, ytrain)#, eval_set = [[xtrain, ytrain],[xvalid, yvalid]])
        
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


    # print(f"model={model_name}, fold={fold}, SMAPE: train {smape_train:.4f}, valid {smape_valid:.4f}")

    fold_dict = {
        # "train_smape" : smape_train,
        "valid_smape" : smape_valid
    }
    return fold_dict


def objective(trial):
    param = {
            'random_seed':42,
            "verbosity": 0,
            "tree_method": 'hist',#'exact',
            "use_label_encoder": False,
            "booster": "gbtree",
            "tree_method":'gpu_hist', 
            "gpu_id":0,

            "eval_metric":trial.suggest_categorical("eval_metric", ['rmse','mape']),
            # 'n_estimators': trial.suggest_categorical("n_estimators", [150, 200, 300, 700, 1000, 1500, 2000]),
            'n_estimators': trial.suggest_int("n_estimators", 100, 2000),
            "max_depth" : trial.suggest_int("max_depth", 2, 8),
            'eta': trial.suggest_float('eta', 1e-5, 0.1),
            # 'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.009,0.01,0.012,0.014,0.016,0.018, 0.02]),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
            "gamma" : trial.suggest_float("gamma", 1e-5, 1.0),
            # 'subsample': trial.suggest_categorical( 'subsample', [0.6,0.7,0.8,1.0]),
            'subsample': trial.suggest_float('subsample', 0.6,1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            # 'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5,0.6,0.7,0.8,0.9,1.0]),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5,1.0),
        }
       
    df = pd.read_csv(os.path.join(config.ROOT_DIR,"data","processed","train_feat_eng_01.csv"))

    folds = 4
    # smape_train_avg = []
    smape_valid_avg = []
    for fold in range(folds):
        fold_dict = train_folds(fold=fold,df=df,param=param)
        # smape_train_avg.append(fold_dict['train_smape'])
        smape_valid_avg.append(fold_dict['valid_smape'])

    # return np.mean(smape_train_avg) - np.mean(smape_valid_avg), np.mean(smape_valid_avg)
    return np.mean(smape_valid_avg)


if __name__ == '__main__':
    study = optuna.create_study(
        study_name='xgb-optim',
        sampler = optuna.samplers.TPESampler(seed=142),
        direction="minimize",
        # directions=["minimize", "minimize"],
        )
    study.optimize(objective, n_trials=500, n_jobs= -1)

    print("best trials:")
    trial_ = study.best_trial

    # print(trial_)
    print(trial_.values)
    print(trial_.params)
    
    logging.basicConfig(
        filename=os.path.join(config.ROOT_DIR,"logs",f"xgb_param.log"),
        filemode='a',
        level=logging.INFO)
    logging.info(f'\n values = {trial_.values} \n param = {trial_.params}')

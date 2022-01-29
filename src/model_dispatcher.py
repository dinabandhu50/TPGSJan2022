import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import config
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

SEED = config.SEED

rf_param = {
    'random_state' : SEED,
    'n_estimators' : 100,
    'max_depth' : 6
}
xgb_param = {
    'random_seed':SEED,
    'verbosity': 0,
    'tree_method': 'hist',
    'use_label_encoder': False,
    'booster': 'gbtree',
    'tree_method':'gpu_hist', 
    'gpu_id':0,
    'eval_metric':'rmse',
    # 'eval_metric':'mape',

    # 'n_estimators': 4000, #2000 
    'early_stopping_rounds' : 200,
    # 'max_depth': 3, 
    # 'eta': 0.01886834650237164, 
    # 'learning_rate': 0.018, 
    # 'colsample_bytree': 0.9, 
    # 'min_child_weight': 3,
    # 'subsample': 0.6, 
    # 'alpha': 3.284395992431614, 
    # 'lambda': 0.0012338191278124635, 

    # 'n_estimators': 2000, 
    # 'subsample': 0.7, 
    # 'colsample_bytree': 0.7, 
    # 'eta': 0.0031426086000654833, 
    # 'reg_alpha': 30, 
    # 'reg_lambda': 17, 
    # 'max_depth': 3, 
    # # 'max_depth': 7, 

    'min_child_weight': 17,
    'n_estimators': 2000, 
    'subsample': 0.6, 
    'max_depth': 5, 
    'eta': 0.09886834650237164, 
    'colsample_bytree': 0.9, 
    'learning_rate': 0.018, 
    'min_child_weight': 3,
    'alpha': 3.284395992431614, 
    'lambda': 0.0012338191278124635, 
    }

{
    'n_estimators': 2000, 
    'subsample': 0.6, 
    'max_depth': 5, 
    'eta': 0.09886834650237164, 
    'colsample_bytree': 0.9, 
    'learning_rate': 0.018, 
    'min_child_weight': 3,
    'alpha': 3.284395992431614, 
    'lambda': 0.0012338191278124635, 
    }


cat_param = {
    'random_seed':SEED,
    'verbose':False,
    'task_type':"GPU",
    'loss_function':'RMSE',
    # 'eval_metric':'RMSE',
    # 'eval_metric':'SMAPE',
    'eval_metric':'MAPE',
    # 'leaf_estimation_method':'Newton',
    # 'bootstrap_type': 'Bernoulli',
    
    'iterations': 1000,
    'early_stopping_rounds':100,
    'depth': 5, 
    'learning_rate': 0.01,
    # 'l2_leaf_reg': 5.0,
    # 'random_strength': 2.0,
    # 'min_data_in_leaf': 2
}

# model dictionary
models = {
    "rf" : RandomForestRegressor(**rf_param,n_jobs=-1),
    "xgb":XGBRegressor(**xgb_param),
    "cat":CatBoostRegressor(**cat_param),
}
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

    'n_estimators': 2000, 
    'early_stopping_rounds' : 200,
    'min_child_weight': 17,
    'subsample': 0.6, 
    'max_depth': 5, 
    'eta': 0.09886834650237164, 
    'colsample_bytree': 0.9, 
    'learning_rate': 0.018, 
    'min_child_weight': 3,
    'alpha': 3.284395992431614, 
    'lambda': 0.0012338191278124635, 
    }
# Trial 35 finished with value: 6.26311056093722 and parameters: 
# xgb_param = {
#     'random_seed':SEED,
#     'verbosity': 0,
#     'tree_method': 'hist',
#     'use_label_encoder': False,
#     'booster': 'gbtree',
#     'tree_method':'gpu_hist', 
#     'gpu_id':0,
#     'eval_metric': 'mape', 

#     'n_estimators': 1000, 
#     'early_stopping_rounds' : 200,
#     'max_depth': 16, 
#     'eta': 0.013687694568588373, 
#     'learning_rate': 0.01, 
#     'gamma': 0.9794733224016161, 
#     'subsample': 0.7, 
#     'min_child_weight': 6, 
#     'lambda': 9.784253450743748, 
#     'alpha': 9.547416093993132, 
#     'colsample_bytree': 0.8,
#     }


cat_param = {
    'random_seed':SEED,
    'verbose':False,
    # 'task_type':"GPU",

    # 'loss_function':'RMSE',
    # # 'eval_metric':'RMSE',
    # 'eval_metric':'SMAPE',
    # # 'eval_metric':'MAPE',
    # # 'leaf_estimation_method':'Newton',
    # # 'bootstrap_type': 'Bernoulli',
    
    # # 'iterations': 1000,
    # # 'early_stopping_rounds':100,
    # # 'depth': 5, 
    # # 'learning_rate': 0.01,
    # # 'l2_leaf_reg': 5.0,
    # # 'random_strength': 2.0,
    # # 'min_data_in_leaf': 2
    
    'iterations': 10000,
    # "early_stopping_rounds": 1000,
    'depth': 5, 
    'l2_leaf_reg': 12.06,
    'bootstrap_type': 'Bayesian',
    'boosting_type': 'Plain',
    'loss_function': 'MAE',
    'eval_metric': 'SMAPE',
    'od_type': 'Iter',       # type of overfitting detector
    'od_wait': 40,
    'has_time': True,  
}


# model dictionary
models = {
    "rf" : RandomForestRegressor(**rf_param,n_jobs=-1),
    "xgb":XGBRegressor(**xgb_param),
    "cat":CatBoostRegressor(**cat_param),
}
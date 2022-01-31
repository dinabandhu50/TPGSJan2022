import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import config
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import VotingRegressor
# from sklearn.ensemble import StackingRegressor


SEED = config.SEED

rf_param = {
    'random_state' : SEED,
    'n_estimators' : 500,
    'max_depth' : 12,
    # 'min_samples_split': 0.001,
    'min_samples_leaf':0.001,
    # 'max_features':'sqrt',
    # 'ccp_alpha':0.000001,
}


xgb_param = {
    'random_seed':SEED,
    'verbosity': 0,
    'tree_method': 'hist',
    'use_label_encoder': False,
    'booster': 'gbtree',
    'tree_method':'gpu_hist', 
    'gpu_id':0,

    # 'eval_metric':'rmse',
#     'n_estimators': 2000, 
#     'early_stopping_rounds' : 200,
#     'min_child_weight': 17,
#     'subsample': 0.6, 
#     'max_depth': 5, 
#     'eta': 0.09886834650237164, 
#     'colsample_bytree': 0.9, 
#     'learning_rate': 0.018, 
#     'min_child_weight': 3,
#     'alpha': 3.284395992431614, 
#     'lambda': 0.0012338191278124635, 
#     }
# {
    'eval_metric': 'mape', 
    'n_estimators': 1090, 
    'max_depth': 4,#5, 
    'eta': 0.07469880966778804, 
    'learning_rate': 0.08789242310145122, 
    'gamma': 0.0003035530433724634, 
    'subsample': 0.933526972999481, 
    'min_child_weight': 177, 
    'lambda': 0.21611334256426223, 
    # 'alpha': 0.002359459696445972, 
    'alpha': 0.02359459696445972, 
    'colsample_bytree': 0.5015988051009448
    }

cat_param = {
    'random_seed':SEED,
    'verbose':False,
    # 'task_type':"GPU",
   
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

meta_rf_param = {
    'random_state' : SEED+273,
    'n_estimators' : 500,
    'max_depth' : 6,
    # 'min_samples_split': 0.01,

    # 'min_samples_leaf':0.001,
    'min_samples_leaf':0.008,

    # 'max_features':'sqrt',
    # 'ccp_alpha':0.000001,
}

# model loading
model1 = CatBoostRegressor(**cat_param)
model2 = XGBRegressor(**xgb_param)
model3 = RandomForestRegressor(**rf_param,n_jobs=-1)
meta_model = RandomForestRegressor(**meta_rf_param,n_jobs=-1)


# stack model define and fitting
stack_param = {
    "regressors": [model1, model2, model3], 
    "meta_regressor": meta_model, 
    "use_features_in_secondary": True,
    # "verbose": 1,
    "verbose": 0,
}
skvote_param = {
    'estimators': [
        ('cat', model1), 
        ('xgb', model2),
        ('rf', model3),
        ],
    'weights':[0.5,0.3,0.2],
    }


# model dictionary
models = {
    "rf" : RandomForestRegressor(**rf_param,n_jobs=-1),
    "xgb":XGBRegressor(**xgb_param),
    "cat":CatBoostRegressor(**cat_param),
    "stack": StackingRegressor(**stack_param),
    "skvote": VotingRegressor(**skvote_param,n_jobs=-1),
}
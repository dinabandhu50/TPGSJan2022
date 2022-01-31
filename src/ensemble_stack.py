import os
import config
import joblib
import numpy as np
import pandas as pd

from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import KFold
from sklearn import metrics

# read data
df = pd.read_csv(config.TRAIN_FOLDS)
feature_cols = df.drop(['loss','id','kfold'],axis=1).columns.tolist()
target_cols = ['loss']

# train and valid split
xtrain = df[feature_cols]
ytrain = df[target_cols].values.ravel()

# k fold define
kfold = KFold(n_splits=10,random_state=42,shuffle=True)

# model loading
model1 = joblib.load(os.path.join(config.MODELS,f"model_cat.bin"))
model2 = joblib.load(os.path.join(config.MODELS,f"model_xgb.bin"))
model3 = joblib.load(os.path.join(config.MODELS,f"model_xgbrf.bin"))
model4 = joblib.load(os.path.join(config.MODELS,f"model_xgb.bin"))


regr_models = (model1,model2,model3,model4)


# stack model define and fitting
model_stack = StackingCVRegressor(regressors=regr_models, meta_regressor=model2, 
                            use_features_in_secondary=True,shuffle=False,cv=kfold,random_state=42,verbose=1)
model_stack.fit(xtrain, ytrain)


# prediction
predicted_st = model_stack.predict(xtrain)

mae = metrics.mean_absolute_error(ytrain, predicted_st)
mse = metrics.mean_squared_error(ytrain, predicted_st)
rmse = np.sqrt(mse) 
r2 = metrics.r2_score(ytrain,predicted_st)

print("mae:",mae)
print("mse:", mse)
print("rmse:", rmse)
print("r2:", r2)

# prepare submission
df_test = pd.read_csv(config.TEST_DATA_RAW)
ss = pd.read_csv(config.SS_PATH)

feature_cols = df_test.drop(["id"],axis=1).columns.tolist()
xtest = df_test[feature_cols]


ytest = model_stack.predict(xtest)

ss.iloc[:,1:] = ytest
ss.to_csv(os.path.join(config.SUB,f"submission_stack_full.csv"),index=False)
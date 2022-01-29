import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import config
import numpy as np
import pandas as pd

import joblib
import time


def predict(model_name):
    df_test = pd.read_csv(config.TEST_DATA__PROCESSED)
    ss = pd.read_csv(config.SS_PATH)

    feature_cols = df_test.drop(columns=['row_id']).columns.tolist()
    xtest = df_test[feature_cols]

    model = joblib.load(os.path.join(config.ROOT_DIR,"models","all", f"{model_name}_all.joblib"))
    ytest = model.predict(xtest)

    ss.iloc[:,1:] = ytest
    ss.to_csv(os.path.join(config.ROOT_DIR,"data","submissions",f"submission_{model_name}.csv"),index=False)
    return ss

if __name__ == '__main__':
    start_time = time.time()

    # model_names = ['rf','xgb','cat']
    model_names = ['xgb']
    # model_names = ['rf']
    # model_names = ['rf','cat']

    
    for model_name in model_names:
        df_sub = predict(model_name=model_name)
    
    print(f"--- {time.time() - start_time:.4f} seconds ---")
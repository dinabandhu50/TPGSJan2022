import os
import config
import numpy as np
import pandas as pd
import holidays

years = [2015, 2016, 2017, 2018, 2019]
country_list = [
    ("Finland", "FI"),
    ("Norway", "NO"),
    ("Sweden", "SE")
]
holiday_dict = {country[0]: holidays.CountryHoliday(country[1], years=years) for country in country_list}

def get_holiday_name(row):
    try:
        return holiday_dict[row["country"]][row["date"]]
    except:
        return "NA"

df_gdp = pd.read_csv(os.path.join(config.ROOT_DIR,"data","others",'GDP_data_2015_to_2019_Finland_Norway_Sweden.csv'), index_col='year')

def engineer(df):
    df = df.copy()
    # converting to time-stamp
    df['date'] = pd.to_datetime(df['date'])
    
    t0 = np.datetime64('2015-01-01')
    df['time_step'] = (df["date"] - t0).astype('timedelta64[D]').astype(np.int)

    df['year'] = df['date'].dt.year
    # quarter
    df['quarter'] = df['date'].dt.quarter
    df['quarter_sin'] = np.sin(df['quarter'] * (2 * np.pi / 4))
    df['quarter_cos'] = np.cos(df['quarter'] * (2 * np.pi / 4))
    # month
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(df['month'] * (2 * np.pi / 12))
    df['month_cos'] = np.cos(df['month'] * (2 * np.pi / 12))
    # week
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['week_sin'] = np.sin(df['week'] * (2 * np.pi / 52))
    df['week_cos'] = np.cos(df['week'] * (2 * np.pi / 52))
    # day
    df['day'] = df['date'].dt.day
    df['day_sin'] = np.sin(df['day'] * (2 * np.pi / 31))
    df['day_cos'] = np.cos(df['day'] * (2 * np.pi / 31))
    # others
    df['day_of_year'] = df['date'].dt.day_of_year
    df['day_of_year_sin'] = np.sin(df['day_of_year'] * (2 * np.pi / 366))
    df['day_of_year_cos'] = np.cos(df['day_of_year'] * (2 * np.pi / 366))
    
    df['day_of_week'] = df['date'].dt.weekday
    df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
    
    df["is_weekend"] = df['day_of_week'] >= 5

    # gdp column
    df_gdp.columns = ['Finland', 'Norway', 'Sweden']
    gdp_dictionary = df_gdp.unstack().to_dict()
    df['gdp'] = df.set_index(['country','year']).index.map(gdp_dictionary.get)

    df['holiday_name'] = df.apply(get_holiday_name, axis=1)
    df['is_holiday'] = np.where(df['holiday_name'] != "NA", 1, 0).astype(np.int)
    # one hot encoding
    df = pd.get_dummies(df, columns=['store', 'country', 'product'])
    # dropping unused
    # df = df.drop(columns=["row_id", "date","holiday_name"])
    df = df.drop(columns=["date","holiday_name"])

    return df

if __name__ == '__main__':
    df_train = pd.read_csv(os.path.join(config.ROOT_DIR,'data','processed','train_folds.csv'))
    df_test = pd.read_csv(os.path.join(config.ROOT_DIR,'data','raw','test.csv'))
    
    # do feature engineering
    df_train = engineer(df=df_train)
    df_test = engineer(df=df_test)
    
    # saving data frame to corresponding location
    df_train.to_csv(os.path.join(config.ROOT_DIR,'data','processed','train_feat_eng_00.csv'),index=False)
    df_test.to_csv(os.path.join(config.ROOT_DIR,'data','processed','test_feat_eng_00.csv'),index=False)

    print('-'*20 + 'Done' + '-'*20)
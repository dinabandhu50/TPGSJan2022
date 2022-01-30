import os
import config
import numpy as np
import pandas as pd
import holidays
import dateutil.easter as easter


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

    # # lag features
    # df['Lag_1'] = df['num_sold'].shift(18)

    # adding step 
    df['step'] = df['date']-df['date'].shift(1)     #shift index and find difference
    zero = np.timedelta64(0, 's')       
    df['step'][0] = np.timedelta64(0, 's')          #change first var from naT to zero
    df['step'] = df['step'].apply(lambda x: x>zero).cumsum()
    df['step^2'] = df['step']**2

    # gdp column
    df_gdp.columns = ['Finland', 'Norway', 'Sweden']
    gdp_dictionary = df_gdp.unstack().to_dict()
    df['gdp'] = df.set_index(['country','year']).index.map(gdp_dictionary.get)
    
    # --------------------------------------------------------------------
    # gdp per capita
    Sweden_ec = {2015:[51545,-.1412],2016:[51965,.0081],2017:[53792,.0351],2018:[54589,.0148],2019:[51687,-.0532]}
    Finland_ec = {2015:[42802,-.1495],2016:[43814,.0236],2017:[46412,.0593],2018:[50038,.0781],2019:[48712,-.0265]}
    Norway_ec = {2015:[74356,-.2336],2016:[70461,-.0524],2017:[75497,0.0715],2018:[82268,.0897],2019:[75826,-.0783]}

    df['GDPperCapita'] = [Sweden_ec[a.year][0] if b =='Sweden' else(Finland_ec[a.year][0] if b =='Finland' else Norway_ec[a.year][0]) for a,b in zip(df.date,df.country)]
    df['GrowthRate']  = [Sweden_ec[a.year][1] if b =='Sweden' else(Finland_ec[a.year][1] if b =='Finland' else Norway_ec[a.year][1]) for a,b in zip(df.date,df.country)]

    # holiday
    df['holiday_name'] = df.apply(get_holiday_name, axis=1)
    df['is_holiday'] = np.where(df['holiday_name'] != "NA", 1, 0).astype(np.int)
    
    # features for fe-01 --------------------------------------------------------------
    gdp_exponent = 1.2121103201489674
    df['gdp'] = df['gdp'] ** gdp_exponent
    # Easter
    easter_date = df.date.apply(lambda date: pd.Timestamp(easter.easter(date.year)))
    df['days_from_easter'] = (df.date - easter_date).dt.days.clip(-5, 65)

    # Last Sunday of May (Mother's Day)
    sun_may_date = df.date.dt.year.map({
        2015: pd.Timestamp(('2015-5-31')),
        2016: pd.Timestamp(('2016-5-29')),
        2017: pd.Timestamp(('2017-5-28')),
        2018: pd.Timestamp(('2018-5-27')),
        2019: pd.Timestamp(('2019-5-26'))
    })
    #new_df['days_from_sun_may'] = (df.date - sun_may_date).dt.days.clip(-1, 9)
    
    # Last Wednesday of June
    wed_june_date = df.date.dt.year.map({
        2015: pd.Timestamp(('2015-06-24')),
        2016: pd.Timestamp(('2016-06-29')),
        2017: pd.Timestamp(('2017-06-28')),
        2018: pd.Timestamp(('2018-06-27')),
        2019: pd.Timestamp(('2019-06-26'))
    })
    df['days_from_wed_jun'] = (df.date - wed_june_date).dt.days.clip(-5, 5)

    # First Sunday of November (second Sunday is Father's Day)
    sun_nov_date = df.date.dt.year.map({
        2015: pd.Timestamp(('2015-11-1')),
        2016: pd.Timestamp(('2016-11-6')),
        2017: pd.Timestamp(('2017-11-5')),
        2018: pd.Timestamp(('2018-11-4')),
        2019: pd.Timestamp(('2019-11-3'))
    })
    df['days_from_sun_nov'] = (df.date - sun_nov_date).dt.days.clip(-1, 9)

    # end -------------------------------------------------------------
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
    df_train.to_csv(os.path.join(config.ROOT_DIR,'data','processed','train_feat_eng_01.csv'),index=False)
    df_test.to_csv(os.path.join(config.ROOT_DIR,'data','processed','test_feat_eng_01.csv'),index=False)

    print('-'*20 + 'Done' + '-'*20)
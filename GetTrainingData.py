import RiverLevels
import Rainfall
import pandas as pd
import random
from datetime import timedelta

def generate_sample_range(df: pd.DataFrame, startday: int, days: int) -> pd.DatetimeIndex:

    if startday + days > len(df.index):
        raise Exception("Start day of %s with %s days added is %s which is out of range" % 
                        (
                            (df.index.min() + timedelta(days=startday)).strftime("%y-%m-%d"), 
                            days, 
                            (df.index.min() + timedelta(days=startday + days - 1)).strftime("%y-%m-%d")
                        ))

    start_date = df.index.min()
    end_date = df.index.max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')

    missing_dates = all_dates.difference(df.index)

    sample_start_date = start_date + timedelta(days=startday)

    sample_dates = pd.date_range(start= sample_start_date, end= sample_start_date + timedelta(days= days - 1))

    if sample_dates.isin(missing_dates).any():
        sample_dates = pd.DatetimeIndex([])

    return sample_dates

def getRainLevelDF():
    level_df = RiverLevels.getHistoricLevelData(RiverLevels.riverData.dargle)
    daily_average_level = level_df["MappedValue"].resample('D').agg('mean')
    rain_df = Rainfall.GetHistoricRainfall(RiverLevels.riverData.dargle)
    rain_df.index = rain_df.index.tz_localize('UTC')

    df = rain_df.join(daily_average_level)
    df.dropna(how='any', inplace = True)
    df.drop('ind', axis=1, inplace = True)
    df["DayOfYear"] = df.index.dayofyear
    df = df[["rain", "DayOfYear", "MappedValue"]]
    return df

def getSample(df: pd.DataFrame, ind: int, length: int) -> pd.DataFrame:
    sample_dates = generate_sample_range(df, ind, length)
    sample_df = df[df.index.isin(sample_dates)]
    return sample_df


import pandas as pd
import Data.riverData as riverData

def GetHistoricRainfall(river: riverData.river) -> pd.DataFrame:
    df = pd.read_csv(river.RainfallCSV, skiprows = 9)
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
    df.set_index('date', inplace=True)
    return df

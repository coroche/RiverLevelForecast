import requests
import pandas as pd
from io import StringIO
import json
from typing import List
import Data.riverData as riverData
from enum import Enum

class GaugeData:
    def __init__(self, data: dict):
        self.stationparameter_name: str = data.get("stationparameter_name")
        self.ts_shortname: str = data.get("ts_shortname")
        self.ts_unitsymbol: str = data.get("ts_unitsymbol")
        self.ts_precision: str = data.get("ts_precision")
        self.rows: str = data.get("rows")
        self.columns: str = data.get("columns")
        self.data: List[str] = data.get("data", [])

class period(Enum):
    day = 1
    week = 2
    month = 3

def getLatestLevelData(river: riverData.river, period: period) -> pd.DataFrame:
    
    station = str(river.stationNum)
    period = period.name
    url = "http://waterlevel.ie/data/" + period + "/" + station + "_0001.csv"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.HTTPError as ex:
        raise SystemExit(ex)
    except requests.exceptions.RequestException as ex:
        raise SystemExit(ex)
   
    df = pd.read_csv(StringIO(response.text), dtype = {"datetime": str, "value": float}, parse_dates=["datetime"])
    return df

def getLatestLevelData2(river: riverData.river, period: period) -> pd.DataFrame:

    station = str(river.stationNum)
    period = period.name
    url = "https://waterlevel.ie/hydro-data/data/internet/stations/0/" + station + "/S/" + period + ".json"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.HTTPError as ex:
        raise SystemExit(ex)
    except requests.exceptions.RequestException as ex:
        raise SystemExit(ex)
    
    data = [GaugeData(x) for x in json.loads(response.text)]
    levelData = [x for x in data if x.ts_shortname == "WEB.Cmd.P-Continuous.Absolute"][0]
    df = pd.DataFrame(levelData.data, columns = levelData.columns.split(","))
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def getHistoricLevelData(river: riverData.river) -> pd.DataFrame:

    df = pd.read_csv(river.LevelCSV, skiprows = 10, parse_dates = ["#Timestamp"], delimiter = ";")
    df.columns = ['Timestamp'] + list(df.columns[1:]) #remove hash from column name
    return df

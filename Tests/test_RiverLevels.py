from RiverLevelsAPI import *

def test_getLatestRiverLevels():
    df = getLatestLevelData(riverData.dargle, period.day)
    assert all([col in df.columns for col in ["datetime", "value"]])
    assert len(df.index) > 0
    assert df["datetime"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"

def test_getHistoricRiverLevels():
    df = getHistoricLevelData(riverData.dargle)
    assert all([col in df.columns for col in ["Timestamp", "Value"]])
    assert len(df.index) > 0
    assert df["Timestamp"].dtype == "datetime64[ns, UTC]"
    assert df["Value"].dtype == "float64"

def test_getLatestRiverLevels2():
    df = getLatestLevelData2(riverData.dargle, period.week)
    assert all([col in df.columns for col in ["Timestamp", "Value"]])
    assert len(df.index) > 0
    assert df["Timestamp"].dtype == "datetime64[ns, UTC]"
    assert df["Value"].dtype == "float64"
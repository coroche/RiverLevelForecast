from RiverLevels import *
import Rainfall

def test_getLatestRiverLevels():
    df = getLatestLevelData(riverData.dargle, period.day)
    assert all([col in df.columns for col in ["value"]])
    assert df.index.name == "datetime"
    assert len(df.index) > 0
    assert df.index.dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"

def test_getHistoricRiverLevels():
    df = getHistoricLevelData(riverData.dargle)
    assert all([col in df.columns for col in ["Value"]])
    assert df.index.name == "Timestamp"
    assert len(df.index) > 0
    assert df.index.dtype == "datetime64[ns, UTC]"
    assert df["Value"].dtype == "float64"

def test_getLatestRiverLevels2():
    df = getLatestLevelData2(riverData.dargle, period.week)
    assert all([col in df.columns for col in ["Value"]])
    assert df.index.name == "Timestamp"
    assert len(df.index) > 0
    assert df.index.dtype == "datetime64[ns, UTC]"
    assert df["Value"].dtype == "float64"

def test_getHistoricRainfall():
    df = Rainfall.GetHistoricRainfall(riverData.dargle)
    assert all([col in df.columns for col in ["date", "rain"]])
    assert len(df.index) > 0
    assert df["date"].dtype == "datetime64[ns]"
    assert df["rain"].dtype == "float64"

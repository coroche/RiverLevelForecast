
class river:
    def __init__(self, data: dict) -> None:
        self.stationNum: int = data.get("stationNum")
        self.LevelCSV: str = data.get("LevelCSV")
        self.RainfallCSV: str = data.get("RainfallCSV")

dargle = river({
    "stationNum": 10051,
    "LevelCSV": "Data\Dargle\DargleLevel.csv",
    "RainfallCSV": "Data\Dargle\SallyGapRainFall.csv"
})

glens = river({
    "stationNum": 25309,
    "LevelCSV": "Data\Glens\GlensLevel.csv",
    "RainfallCSV": ""
})
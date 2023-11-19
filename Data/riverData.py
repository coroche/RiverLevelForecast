
class river:
    def __init__(self, data: dict) -> None:
        self.stationNum: int = data.get("stationNum")
        self.CSVLocation: str = data.get("CSVLocation")

dargle = river({
    "stationNum": 10051,
    "CSVLocation": "Data\Dargle\Dargle.csv"
})

glens = river({
    "stationNum": 25309,
    "CSVLocation": "Data\Glens\Glens.csv"
})
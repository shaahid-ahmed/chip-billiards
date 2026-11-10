"""Download missing event CSVs to BDB_DATA, or ./data if it isn't set."""
import os, urllib.request, concurrent.futures as cf
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BDB_DATA", os.path.join(HERE, "data"))
RAW = "https://raw.githubusercontent.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data/main/data/"


def ensure_data():
    os.makedirs(os.path.join(DATA, "tracking"), exist_ok=True)
    for f in ["games.csv", "plays.csv", "players.csv", "pffScoutingData.csv"]:
        if not os.path.exists(os.path.join(DATA, f)):
            urllib.request.urlretrieve(RAW + f, os.path.join(DATA, f))
    games = pd.read_csv(os.path.join(DATA, "games.csv"))
    files = [f"tracking/tracking_{g}.csv" for g in games.gameId]
    miss = [f for f in files if not os.path.exists(os.path.join(DATA, f))]

    def get(f):
        tmp = os.path.join(DATA, f + ".part")
        urllib.request.urlretrieve(RAW + f, tmp)
        os.replace(tmp, os.path.join(DATA, f))

    if miss:
        print(f"downloading {len(miss)} tracking files ...", flush=True)
        with cf.ThreadPoolExecutor(24) as ex:
            list(ex.map(get, miss))
    print("data ready:", DATA, flush=True)


if __name__ == "__main__":
    ensure_data()

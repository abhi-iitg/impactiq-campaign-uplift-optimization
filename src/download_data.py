"""
Download the Hillstrom MineThatData email dataset into data/raw/.
Usage: python src/download_data.py
"""
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT = RAW / "hillstrom.csv"


def main():
    try:
        from sklift.datasets import fetch_hillstrom
        bunch = fetch_hillstrom(target_col="visit")
        df = bunch.data.copy()
        df["segment"] = bunch.treatment
        df["visit"] = bunch.target
        df["conversion"] = fetch_hillstrom(target_col="conversion").target
        df["spend"] = fetch_hillstrom(target_col="spend").target
        df.to_csv(OUT, index=False)
        print(f"Saved {len(df):,} rows -> {OUT}")
    except Exception as e:
        print("Library fetch failed:", e)
        print("Fallback: download 'Hillstrom.csv' from Kaggle "
              "(search 'Kevin Hillstrom MineThatData E-Mail Analytics') "
              f"and place it at {OUT}")


if __name__ == "__main__":
    main()

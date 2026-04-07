import argparse
import json
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

URL_DA = "https://www.omie.es/es/file-download?parents=marginalpdbc&filename=marginalpdbc_{date}.1"
ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "data"

SOLAR_PROFILE = np.array(
    [
        0, 0, 0, 0, 0, 0,
        0.0008, 0.0128, 0.0385, 0.0689, 0.0978, 0.1209,
        0.1343, 0.1357, 0.1247, 0.1038, 0.0764, 0.04767,
        0.02624, 0.0102, 0.0008, 0, 0, 0,
    ],
    dtype=float,
)


def _last_sunday(year: int, month: int) -> date:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    d = next_month - timedelta(days=1)
    while d.weekday() != 6:  # Sunday
        d -= timedelta(days=1)
    return d


def cest_start(year: int) -> date:
    return _last_sunday(year, 3)


def cest_end(year: int) -> date:
    return _last_sunday(year, 10)


def _is_summer_time(d: date) -> bool:
    return cest_start(d.year) <= d < cest_end(d.year)


def _download_text(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text
    except requests.RequestException:
        return None


def _clean_numeric(value: str) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip().replace(",", ".")
    if s.startswith("."):
        s = f"0{s}"
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _strip_header_footer(lines: list[str]) -> list[str]:
    if lines and lines[0].strip().upper().startswith("MARGINAL"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("*"):
        lines = lines[:-1]
    return lines


def _parse_raw_to_periods(raw_text: str) -> pd.DataFrame:
    if not raw_text or raw_text.strip() == "":
        return pd.DataFrame(columns=["date", "period", "DA_ES_PRICE", "DA_PT_PRICE"])

    lines = _strip_header_footer(raw_text.strip().splitlines())
    if not lines:
        return pd.DataFrame(columns=["date", "period", "DA_ES_PRICE", "DA_PT_PRICE"])

    df = pd.read_csv(StringIO("\n".join(lines)), sep=";", header=None, engine="python", dtype=str)
    if df.shape[1] < 6:
        return pd.DataFrame(columns=["date", "period", "DA_ES_PRICE", "DA_PT_PRICE"])

    df = df.iloc[:, :6].copy()
    df.columns = ["year", "month", "day", "period", "DA_PT_PRICE", "DA_ES_PRICE"]

    mask = (
        df["year"].str.isdigit()
        & df["month"].str.isdigit()
        & df["day"].str.isdigit()
        & df["period"].str.isdigit()
    )
    df = df[mask].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "period", "DA_ES_PRICE", "DA_PT_PRICE"])

    df["date"] = pd.to_datetime(
        dict(
            year=df["year"].astype(int),
            month=df["month"].astype(int),
            day=df["day"].astype(int),
        )
    )
    df["period"] = df["period"].astype(int)
    df["DA_ES_PRICE"] = df["DA_ES_PRICE"].apply(_clean_numeric)
    df["DA_PT_PRICE"] = df["DA_PT_PRICE"].apply(_clean_numeric)

    return (
        df[["date", "period", "DA_ES_PRICE", "DA_PT_PRICE"]]
        .sort_values(["date", "period"])
        .reset_index(drop=True)
    )


def _periods_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "hour", "DA_ES_PRICE", "DA_PT_PRICE"])

    pmax = int(df["period"].max())
    if pmax > 30:
        df = df.copy()
        df["hour"] = ((df["period"] - 1) // 4) + 1
        out = (
            df.groupby(["date", "hour"], as_index=False)[["DA_ES_PRICE", "DA_PT_PRICE"]]
            .mean()
            .sort_values(["date", "hour"])
            .reset_index(drop=True)
        )
    else:
        out = (
            df.rename(columns={"period": "hour"})[["date", "hour", "DA_ES_PRICE", "DA_PT_PRICE"]]
            .sort_values(["date", "hour"])
            .reset_index(drop=True)
        )

    out[["DA_ES_PRICE", "DA_PT_PRICE"]] = out[["DA_ES_PRICE", "DA_PT_PRICE"]].round(2)
    return out


def _to_winter_time(df_today: pd.DataFrame, df_next: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    d = pd.to_datetime(df_today["date"].iloc[0]).date()
    spring_forward = cest_start(d.year)
    fall_back = cest_end(d.year)

    if d == spring_forward:
        date_val = df_today["date"].iloc[0]
        skeleton = pd.DataFrame({"date": date_val, "hour": range(1, 25)})
        df_merged = skeleton.merge(df_today, on=["date", "hour"], how="left")
        df_merged[["DA_ES_PRICE", "DA_PT_PRICE"]] = df_merged[["DA_ES_PRICE", "DA_PT_PRICE"]].ffill()
        return df_merged.reset_index(drop=True)

    if d == fall_back:
        return df_today[df_today["hour"] <= 24].reset_index(drop=True)

    if _is_summer_time(d):
        if df_next is None or df_next.empty:
            raise ValueError(f"Next day's data required to convert {d} from CEST to CET.")

        today_shifted = df_today[df_today["hour"].between(2, 24)].copy()
        today_shifted["hour"] = today_shifted["hour"] - 1

        next_h1 = df_next[df_next["hour"] == 1].copy()
        if next_h1.empty:
            raise ValueError(f"Could not retrieve next day's first hour for {d}.")
        next_h1["hour"] = 24
        next_h1["date"] = df_today["date"].iloc[0]

        return pd.concat([today_shifted, next_h1], ignore_index=True).sort_values("hour").reset_index(drop=True)

    return df_today.reset_index(drop=True)


def download_da_es_pt(date_str: str) -> pd.DataFrame:
    d = pd.to_datetime(date_str, format="%Y%m%d").date()

    raw_today = _download_text(URL_DA.format(date=date_str))
    if raw_today is None:
        return pd.DataFrame(columns=["date", "hour", "DA_ES_PRICE", "DA_PT_PRICE"])

    df_today = _periods_to_hourly(_parse_raw_to_periods(raw_today))
    if df_today.empty:
        return df_today

    df_next = None
    if _is_summer_time(d) and d != cest_end(d.year):
        next_str = (d + timedelta(days=1)).strftime("%Y%m%d")
        raw_next = _download_text(URL_DA.format(date=next_str))
        if raw_next:
            df_next = _periods_to_hourly(_parse_raw_to_periods(raw_next))

    out = _to_winter_time(df_today, df_next)
    out[["DA_ES_PRICE", "DA_PT_PRICE"]] = out[["DA_ES_PRICE", "DA_PT_PRICE"]].round(2)
    return out


def calc_top_bottom_spread(df: pd.DataFrame, hours: int) -> dict:
    date_str = pd.to_datetime(df["date"].iloc[0]).strftime("%Y%m%d")
    if len(df) < 2 * hours:
        raise ValueError(f"Need at least {2 * hours} hours of data for a {hours}h product, got {len(df)}")

    results = {"date": date_str}
    for price_col, key in [("DA_ES_PRICE", "TB_ES"), ("DA_PT_PRICE", "TB_PT")]:
        sorted_prices = df[price_col].sort_values()
        bottom_h = sorted_prices.iloc[:hours]
        top_h = sorted_prices.iloc[-hours:]
        spread = top_h.sum() - bottom_h.sum()
        results[key] = float(round(spread / hours, 2))
    return results


def calc_solar_capture(df: pd.DataFrame) -> dict:
    date_str = pd.to_datetime(df["date"].iloc[0]).strftime("%Y%m%d")
    if len(df) != 24:
        raise ValueError(f"Expected 24 hours after winter-time conversion, got {len(df)}")

    weights = SOLAR_PROFILE / SOLAR_PROFILE.sum()
    results = {"date": date_str}
    for price_col, key in [("DA_ES_PRICE", "CAP_ES"), ("DA_PT_PRICE", "CAP_PT")]:
        prices = df[price_col].to_numpy(dtype=float)
        results[key] = float(round(np.dot(weights, prices), 2))
    return results


def calc_avg_price(df: pd.DataFrame) -> dict:
    date_str = pd.to_datetime(df["date"].iloc[0]).strftime("%Y%m%d")
    return {
        "date": date_str,
        "AVG_ES": float(round(df["DA_ES_PRICE"].mean(), 2)),
        "AVG_PT": float(round(df["DA_PT_PRICE"].mean(), 2)),
    }


def _load_json_records(filepath: Path) -> list:
    if not filepath.exists():
        return []
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json_records(filepath: Path, records: list) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")


def calc_and_save_top_bottom(date_str: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = download_da_es_pt(date_str)
    if df.empty:
        raise ValueError(f"No data available for {date_str}")

    for hours in [2, 4, 6]:
        result = calc_top_bottom_spread(df, hours)
        for zone, key in [("ES", "TB_ES"), ("PT", "TB_PT")]:
            filepath = OUT_DIR / f"TB_{zone}_{hours}h.json"
            records = _load_json_records(filepath)
            records = [r for r in records if r["date"] != date_str]
            records.append({"date": date_str, "value": result[key]})
            records.sort(key=lambda r: r["date"])
            _save_json_records(filepath, records)
            print(f"Saved: {filepath.relative_to(ROOT_DIR)}")


def calc_and_save_prices(date_str: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = download_da_es_pt(date_str)
    if df.empty:
        raise ValueError(f"No data available for {date_str}")

    avg = calc_avg_price(df)
    capture = calc_solar_capture(df)

    for zone, avg_key, cap_key in [("ES", "AVG_ES", "CAP_ES"), ("PT", "AVG_PT", "CAP_PT")]:
        filepath = OUT_DIR / f"PRICES_{zone}.json"
        records = _load_json_records(filepath)
        records = [r for r in records if r["date"] != date_str]
        records.append({"date": date_str, "avg": avg[avg_key], "solar_capture": capture[cap_key]})
        records.sort(key=lambda r: r["date"])
        _save_json_records(filepath, records)
        print(f"Saved: {filepath.relative_to(ROOT_DIR)}")


def run_for_date(date_str: str) -> None:
    calc_and_save_top_bottom(date_str)
    calc_and_save_prices(date_str)


def run_for_range(start_str: str, end_str: str) -> None:
    start = datetime.strptime(start_str, "%Y%m%d").date()
    end = datetime.strptime(end_str, "%Y%m%d").date()
    if start > end:
        raise ValueError(f"start ({start_str}) must be <= end ({end_str})")

    d = start
    while d <= end:
        current = d.strftime("%Y%m%d")
        try:
            run_for_date(current)
        except Exception as exc:
            print(f"Skipped {current}: {exc}")
        d += timedelta(days=1)


def yesterday_madrid() -> str:
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Europe/Madrid"))
    except Exception:
        now = datetime.utcnow()
    return (now.date() - timedelta(days=1)).strftime("%Y%m%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OMIE DA data and update Top-Bottom JSON files.")
    parser.add_argument("--date", help="Single date in YYYYMMDD format.")
    parser.add_argument("--start", help="Range start in YYYYMMDD format.")
    parser.add_argument("--end", help="Range end in YYYYMMDD format.")
    parser.add_argument(
        "--yesterday",
        action="store_true",
        help="Process D-1 using Europe/Madrid timezone.",
    )
    args = parser.parse_args()

    if args.start or args.end:
        if not (args.start and args.end):
            raise ValueError("Both --start and --end are required for a range run.")
        run_for_range(args.start, args.end)
        return

    if args.date:
        run_for_date(args.date)
        return

    if args.yesterday or (not args.date and not args.start and not args.end):
        run_for_date(yesterday_madrid())
        return


if __name__ == "__main__":
    main()

"""Source adapter: copy VN30F1M 1-minute bars from PostgreSQL into immutable v0.5-format day files.

Provider convention, verified on 2026-09-25 against the v0.5 CSV (39,521 bars, 2026-01-02..2026-09-04):
- `bars_1m.ts` is timestamptz; converted to Asia/Ho_Chi_Minh it equals the CSV TRADING_DATE +
  TRADING_TIME exactly (OHLC identical, no minute shift). The label is the bar START: the 14:27
  bar is written at ~14:28:02 (`updated_at`), i.e. right after it closes.
- `buy_vol/buy_val` and `sell_vol/sell_val` are swapped relative to the CSV. ARSH does not use
  them; they are written in the CSV orientation so day files keep the v0.5 format.
- A day is complete when its 14:45 ATC bar exists and every bar has `is_final = true`.

Each day is written once to data/days/VN30F1M_<date>_v<N>.csv with a JSON sidecar. If the
provider later revises a day, a new version is written; old files are never overwritten.
Credentials come from the PG_DSN environment variable only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import psycopg

HERE = Path(__file__).resolve().parent
DAYS_DIR = HERE / "data" / "days"
SYMBOL = "VN30F1M"
MARKET_TZ = "Asia/Ho_Chi_Minh"
ATC_TIME = time(14, 45)
HEADER = ["SYMBOL", "TRADING_DATE", "OPEN_PX", "HIGH_PX", "LOW_PX", "CLOSE_PX", "VOL",
          "TRADING_TIME", "BUY_VOL", "BUY_VAL", "SELL_VOL", "SELL_VAL"]
CONTINUOUS = ((time(9, 0), time(11, 29)), (time(13, 0), time(14, 29)))

QUERY = f"""
select (ts at time zone '{MARKET_TZ}') as local_ts, open, high, low, close, volume,
       buy_vol, buy_val, sell_vol, sell_val, is_final,
       (updated_at at time zone '{MARKET_TZ}') as local_updated
from bars_1m
where symbol = %s and ts >= (%s::timestamp at time zone '{MARKET_TZ}')
      and ts < (%s::timestamp at time zone '{MARKET_TZ}')
order by ts
"""


def number(value) -> str:
    """Render like the v0.5 CSV: prices keep one decimal (1930.0), integral values stay integers."""
    if value is None:
        return ""
    as_float = float(value)
    return repr(as_float)


def integer(value) -> str:
    return "" if value is None else str(int(round(float(value))))


def expected_minutes(day: date) -> list[datetime]:
    minutes = []
    for start, end in CONTINUOUS:
        cursor = datetime.combine(day, start)
        while cursor.time() <= end:
            minutes.append(cursor)
            cursor += timedelta(minutes=1)
    return minutes


def fetch_rows(conn, start: date, end: date) -> dict[date, list[tuple]]:
    by_day: dict[date, list[tuple]] = {}
    with conn.cursor() as cur:
        cur.execute(QUERY, (SYMBOL, start, end + timedelta(days=1)))
        for row in cur.fetchall():
            by_day.setdefault(row[0].date(), []).append(row)
    return by_day


def render_day(rows: list[tuple]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    for (ts, open_, high, low, close, volume, buy_vol, buy_val, sell_vol, sell_val, _, _) in rows:
        # CSV orientation: BUY_* <- sell_*, SELL_* <- buy_* (see module docstring).
        writer.writerow([SYMBOL, ts.strftime("%Y%m%d"), number(open_), number(high), number(low),
                         number(close), integer(volume), ts.strftime("%H:%M:%S"),
                         integer(sell_vol), integer(sell_val), integer(buy_vol), integer(buy_val)])
    return buffer.getvalue()


def audit_day(day: date, rows: list[tuple]) -> dict:
    stamps = {row[0].replace(tzinfo=None) for row in rows}
    missing = [m.strftime("%H:%M") for m in expected_minutes(day) if m not in stamps]
    has_atc = any(row[0].time() == ATC_TIME for row in rows)
    all_final = all(bool(row[10]) for row in rows)
    warnings = []
    if missing:
        warnings.append(f"missing_continuous_minutes:{','.join(missing)}")
    if not has_atc:
        warnings.append("missing_atc_bar")
    if not all_final:
        warnings.append("bars_not_final")
    return {
        "complete": has_atc and all_final,
        "bars": len(rows),
        "expected_continuous_bars": len(expected_minutes(day)),
        "missing_continuous_minutes": missing,
        "available_at": max(row[11] for row in rows).isoformat(),
        "warnings": warnings,
    }


def latest_version(day: date) -> tuple[int, dict | None]:
    versions = sorted(DAYS_DIR.glob(f"{SYMBOL}_{day}_v*.json"),
                      key=lambda path: int(path.stem.rsplit("_v", 1)[1]))
    if not versions:
        return 0, None
    last = versions[-1]
    return int(last.stem.rsplit("_v", 1)[1]), json.loads(last.read_text(encoding="utf-8"))


def store_day(day: date, rows: list[tuple]) -> dict:
    """Write the day if new or revised; return the sidecar of the current version."""
    text = render_day(rows)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    audit = audit_day(day, rows)
    version, previous = latest_version(day)
    if not audit["complete"]:
        # Only complete days become immutable files; an unfinished day is reported, not stored.
        return {"symbol": SYMBOL, "date": str(day), "version": None, "stored": False, **audit}
    if previous is not None and previous["sha256"] == digest:
        return previous
    version += 1
    stem = f"{SYMBOL}_{day}_v{version}"
    csv_path = DAYS_DIR / f"{stem}.csv"
    meta = {
        "symbol": SYMBOL, "date": str(day), "version": version, "stored": True,
        "revises_version": version - 1 if previous else None,
        "csv": csv_path.name, "sha256": digest,
        "source": "postgresql bars_1m via PG_DSN",
        "timestamp_convention": "bar start, Asia/Ho_Chi_Minh; verified equal to v0.5 CSV on 2026-01-02..2026-09-04",
        "completeness_rule": "14:45 ATC bar present and all bars is_final",
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **audit,
    }
    DAYS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(text, encoding="utf-8", newline="")
    (DAYS_DIR / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    return meta


def market_activity(day: date) -> dict:
    """Bars seen on `day` for VN30F1M and for other Vietnamese symbols.

    `web.trading_calendar` is not usable: it marks every weekday as a session, including Tet,
    30/4 and 2/9. Instead, a day is a session if any Vietnamese symbol has bars (on 2/9/2026
    none did; on normal days VNINDEX and ~30 stocks do). Global feeds (prefix `G-`) are ignored.
    """
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        raise SystemExit("PG_DSN is not set")
    sql = f"""select count(*) filter (where symbol = %s), count(distinct symbol) filter (where symbol <> %s)
              from bars_1m where symbol not like 'G-%%'
                and ts >= (%s::timestamp at time zone '{MARKET_TZ}')
                and ts < (%s::timestamp at time zone '{MARKET_TZ}')"""
    with psycopg.connect(dsn, connect_timeout=15) as conn, conn.cursor() as cur:
        cur.execute(sql, (SYMBOL, SYMBOL, day, day + timedelta(days=1)))
        own, others = cur.fetchone()
    return {"vn30f1m_bars": int(own), "other_vn_symbols": int(others)}


def sync(start: date, end: date) -> list[dict]:
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        raise SystemExit("PG_DSN is not set")
    with psycopg.connect(dsn, connect_timeout=15) as conn:
        by_day = fetch_rows(conn, start, end)
    return [store_day(day, by_day[day]) for day in sorted(by_day)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    for meta in sync(args.start, args.end):
        flag = f"v{meta['version']} stored" if meta["stored"] else "INCOMPLETE, not stored"
        print(f"{meta['date']} {meta['bars']} bars {flag} {' '.join(meta['warnings'])}")


if __name__ == "__main__":
    main()

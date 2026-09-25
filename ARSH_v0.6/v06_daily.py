"""Daily driver: sync complete days from the database, then write EOD reports for days not yet done.

Only new or revised days are processed (tracked in outputs/eod/processed.json by source hash).
The model is never refitted here.

Handling of the current day (scheduled run at 15:15):
- Saturday/Sunday, or a date in market_holidays.txt: no session, no report.
- Session detected (VN30F1M or other Vietnamese symbols have bars) but VN30F1M not complete:
  wait and re-check every --poll-minutes until --wait-until; if still incomplete, write
  pending_<date>.json and exit with code 2. The day is retried automatically on the next run.
- No Vietnamese bars at all on a weekday: holiday or source outage cannot be told apart, so it
  is recorded as `no_vn_data_unverified` (never as "no new day") after waiting the same way.
Past days are always re-synced, so a day that completes late is reported by the next run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time as clock
from datetime import date, datetime, time
from pathlib import Path

import v06_fetch_db as fetch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "eod"
LEDGER = OUT / "processed.json"
LOG = OUT / "daily_log.txt"
HOLIDAYS = HERE / "market_holidays.txt"
MODEL_FIT_CUTOFF = date(2025, 11, 6)  # default v0.5 artifact was fitted on data before this day


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def known_holidays() -> set[date]:
    if not HOLIDAYS.is_file():
        return set()
    days = set()
    for line in HOLIDAYS.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            days.add(date.fromisoformat(text))
    return days


def write_status(day: date, status: str, detail: dict) -> None:
    (OUT / f"pending_{day}.json").write_text(json.dumps(
        {"date": str(day), "status": status, "checked_at": datetime.now().isoformat(timespec="seconds"),
         **detail}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sync_and_report(repo: Path, start: date, end: date, extra: list[str]) -> tuple[dict, int]:
    """Store complete days, report the ones not yet reported. Returns (ledger, failures)."""
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.is_file() else {}
    failed = 0
    for meta in fetch.sync(start, end):
        day = meta["date"]
        if not meta["stored"]:
            continue  # incomplete: handled by the caller, never stored or reported
        if ledger.get(day, {}).get("source_sha256") == meta["sha256"]:
            continue
        sidecar = fetch.DAYS_DIR / meta["csv"].replace(".csv", ".json")
        result = subprocess.run(
            [sys.executable, str(HERE / "v06_replay_eod.py"), "--repo", str(repo),
             "--data", str(fetch.DAYS_DIR / meta["csv"]), "--date", day,
             "--source-meta", str(sidecar), "--out", str(OUT), *extra],
            capture_output=True, text=True)
        if result.returncode != 0:
            tail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
            log(f"{day}: LOI khi lap bao cao - {tail}")
            failed += 1
            continue
        report = next(line.split(": ", 1)[1] for line in result.stdout.splitlines()
                      if line.startswith("End-of-day report"))
        ledger[day] = {"source_version": meta["version"], "source_sha256": meta["sha256"],
                       "report": Path(report).name}
        (OUT / f"pending_{day}.json").unlink(missing_ok=True)
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        warn = f" | canh bao: {' '.join(meta['warnings'])}" if meta["warnings"] else ""
        log(f"{day}: da lap bao cao (ban du lieu v{meta['version']}){warn}")
    return ledger, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=MODEL_FIT_CUTOFF)
    parser.add_argument("--wait-until", type=time.fromisoformat, default=time(18, 0),
                        help="keep re-checking today's data until this local time")
    parser.add_argument("--poll-minutes", type=float, default=10.0)
    parser.add_argument("--shard", type=Path, help="model shard passed to v06_replay_eod.py")
    parser.add_argument("--today", type=date.fromisoformat, default=date.today(),
                        help="treat this date as the current day (manual re-check or testing)")
    parser.add_argument("--out", type=Path, help="report folder; keep one folder per model")
    args = parser.parse_args()
    extra = ["--shard", str(args.shard)] if args.shard else []
    if args.out:
        global OUT, LEDGER, LOG
        OUT = args.out.resolve()
        LEDGER, LOG = OUT / "processed.json", OUT / "daily_log.txt"

    OUT.mkdir(parents=True, exist_ok=True)
    today = args.today
    log(f"Bat dau chay cho ngay {today}")
    while True:
        ledger, failed = sync_and_report(args.repo, args.start, today, extra)
        if failed:
            log(f"{failed} ngay bi loi khi lap bao cao")
            sys.exit(1)
        if str(today) in ledger:
            log(f"{today}: du lieu da hoan tat, bao cao da lap")
            return
        if today.weekday() >= 5 or today in known_holidays():
            log(f"{today}: khong co phien giao dich - khong lap bao cao")
            return
        activity = fetch.market_activity(today)
        session = activity["vn30f1m_bars"] > 0 or activity["other_vn_symbols"] > 0
        status = "session_incomplete" if session else "no_vn_data_unverified"
        if datetime.now().time() >= args.wait_until:
            write_status(today, status, activity)
            if session:
                log(f"{today}: co phien nhung den {args.wait_until:%H:%M} du lieu VN30F1M van chua hoan tat "
                    f"({activity['vn30f1m_bars']} nen) - chua lap bao cao, se thu lai o lan chay sau")
                sys.exit(2)
            log(f"{today}: den {args.wait_until:%H:%M} khong co du lieu ma Viet Nam nao - "
                "co the la ngay nghi hoac nguon loi (chua xac dinh); khong lap bao cao")
            return
        what = (f"co phien, VN30F1M moi co {activity['vn30f1m_bars']} nen" if session
                else "chua thay du lieu ma Viet Nam nao")
        log(f"{today}: {what} - cho {args.poll_minutes:g} phut roi kiem tra lai")
        clock.sleep(args.poll_minutes * 60)


if __name__ == "__main__":
    main()

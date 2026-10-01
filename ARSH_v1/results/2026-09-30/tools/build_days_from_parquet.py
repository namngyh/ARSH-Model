"""Read-only: build per-day VN30F1M CSVs (v0.5 format) from ingestion Parquet bars, 2026-09-07 onward."""
import glob, hashlib, json, os, re
import pandas as pd
SRC = "D:/Database - QuantPercent/data/raw/bars_1m"
OUT = "data/days_from_parquet"
START = "2026-09-05"
def natkey(p):
    n = os.path.basename(p); m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(?:\.part(\d+))?\.parquet", n)
    return (m.group(1), int(m.group(2) or 0)) if m else (n, 0)
files = sorted(glob.glob(SRC + "/*.parquet"), key=natkey)
frames, bad, used = [], [], []
for order, f in enumerate(files):
    if natkey(f)[0] < START: continue
    try:
        d = pd.read_parquet(f, columns=["symbol","ts","open","high","low","close","volume"])
    except Exception as e:
        bad.append(os.path.basename(f)); continue
    d = d[d.symbol == "VN30F1M"].copy(); d["_order"] = order; d["_file"] = os.path.basename(f)
    frames.append(d); used.append(os.path.basename(f))
d = pd.concat(frames)
d["vn"] = pd.to_datetime(d.ts, utc=True).dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)
d = d[d.vn >= START]
for c in ("open","high","low","close","volume"): d[c] = pd.to_numeric(d[c])
g = d.groupby("vn")
conf = g[["open","high","low","close","volume"]].nunique().gt(1).any(axis=1)
conflicts = []
for ts in conf[conf].index:
    rows = d[d.vn == ts][["_file","open","high","low","close","volume"]]
    conflicts.append({"ts": str(ts), "rows": rows.astype(str).values.tolist()})
# rule: within a timestamp keep max volume (bar accumulates during the minute), tie -> latest file order
u = d.sort_values(["vn","volume","_order"]).drop_duplicates("vn", keep="last").sort_values("vn")
days = {}
for day, s in u.groupby(u.vn.dt.date):
    out = pd.DataFrame({"SYMBOL":"VN30F1M","TRADING_DATE":s.vn.dt.strftime("%Y%m%d"),"TRADING_TIME":s.vn.dt.strftime("%H:%M:%S"),
        "OPEN_PX":s.open.values,"HIGH_PX":s.high.values,"LOW_PX":s.low.values,"CLOSE_PX":s.close.values,"VOL":s.volume.astype("int64").values})
    path = f"{OUT}/VN30F1M_{day}_v1.csv"; out.to_csv(path, index=False)
    days[str(day)] = {"rows": len(out), "first": out.TRADING_TIME.iloc[0], "last": out.TRADING_TIME.iloc[-1],
                      "sha256": hashlib.sha256(open(path,"rb").read()).hexdigest()}
bdays = [str(x.date()) for x in pd.bdate_range("2026-09-07","2026-09-30")]
summary = {"built_at": pd.Timestamp.now().isoformat(timespec="seconds"), "rule_for_duplicate_timestamps": "max volume, tie -> latest file",
           "files_read": len(used), "unreadable_files": bad, "days": days, "weekdays_without_day_file": [x for x in bdays if x not in days],
           "conflicting_timestamps": conflicts}
json.dump(summary, open("data/days_from_parquet_manifest.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k:v for k,v in summary.items() if k!="days"}, ensure_ascii=False, indent=1)); print({k:v["rows"] for k,v in days.items()})

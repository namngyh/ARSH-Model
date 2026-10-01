import pandas as pd, sys
for name, p in [("laplace","C:/Users/Admin/Desktop/laplace/ohlc_export.csv"),("modus13","D:/systemModus-ver_1.3/data/raw/ohlc_export.csv")]:
    d = pd.read_csv(p, dtype={"TRADING_DATE":str})
    d = d[d.SYMBOL=="VN30F1M"]
    d["dt"]=pd.to_datetime(d.TRADING_DATE+" "+d.TRADING_TIME)
    s = d[(d.dt>="2023-02-01")&(d.dt<"2026-08-01")]
    per = s.groupby(s.dt.dt.normalize()).size()
    print(name, "all rows", len(d), d.dt.min(), d.dt.max())
    print("  2023-02-01..2026-07-31 rows", len(s), "days", per.size, "first", s.dt.min(), "last", s.dt.max())
    print("  bars/day value counts (top):", per.value_counts().head(6).to_dict())
    post = d[d.dt>="2026-08-01"]
    pp = post.groupby(post.dt.dt.normalize()).size()
    print("  post-2026-08-01 days", pp.size, "last", pp.index.max(), "counts", pp.tail(5).to_dict())

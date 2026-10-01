import pandas as pd, hashlib
src = "C:/Users/Admin/Desktop/laplace/ohlc_export.csv"
d = pd.read_csv(src, dtype=str)
d = d[d.SYMBOL=="VN30F1M"]
keep = d[d.TRADING_DATE>="20230201"]
keep.to_csv("data/vn30f1m_20230201_20260904.csv", index=False)
up = keep[keep.TRADING_DATE<="20260731"]
m = pd.read_csv("D:/systemModus-ver_1.3/data/raw/ohlc_export.csv", dtype=str)
m = m[(m.SYMBOL=="VN30F1M")&(m.TRADING_DATE>="20230201")&(m.TRADING_DATE<="20260731")]
cols=["SYMBOL","TRADING_DATE","TRADING_TIME","OPEN_PX","HIGH_PX","LOW_PX","CLOSE_PX","VOL"]
h=lambda x: hashlib.sha256(x[cols].reset_index(drop=True).to_csv(index=False).encode()).hexdigest()[:16]
print("laplace<=0731", len(up), h(up), "modus13<=0731", len(m), h(m), "identical", up[cols].reset_index(drop=True).equals(m[cols].reset_index(drop=True)))
# days excluded if only 241-243 bar days: which days are anomalous
per = up.groupby("TRADING_DATE").size()
print("anomalous days:", per[~per.isin([241,242,243])].to_dict())
print("sum 243/242/241 excluded? total", per.sum())

# common.py
import numpy as np
import pandas as pd

BARS_PER_DAY = 78
CANDIDATE_FACTORS = [
    "zprice_6","zprice_12","zprice_18","zprice_24","zprice_30","zprice_36",
    "zprice_42","zprice_48","zprice_54","zprice_60",
    "volume_adjusted_return_12","volume_adjusted_return_24","volume_adjusted_return_36",
    "volume_adjusted_return_48","volume_adjusted_return_60",
    "volume_adjusted_return_72","volume_adjusted_return_84","volume_adjusted_return_96",
    "volume_adjusted_return_108","volume_adjusted_return_120",
    "volume_adjusted_return_132","volume_adjusted_return_144",
    "volume_weighted_momentum_12","rsi_signal_12","vwap_dev_39"
]

DEFAULT_FACTORS = ["zprice_12","zprice_36","volume_weighted_momentum_12","volume_adjusted_return_12"]


def max_window_from_names(names):
    m = 12
    for name in names:
        for tok in name.split("_"):
            if tok.isdigit():
                m = max(m, int(tok))
    return m


def extract_field_panel(df, field):
    """完全原版 _extract_field_panel，未改动"""
    if isinstance(df.columns, pd.MultiIndex):
        if "field" in df.columns.names:
            out = df.xs(field, axis=1, level="field")
            out.columns = out.columns.astype(str)
            return out.sort_index()
        for lvl in range(df.columns.nlevels):
            vals = [str(x).lower() for x in df.columns.get_level_values(lvl)]
            if field.lower() in vals:
                out = df.xs(field, axis=1, level=lvl)
                out.columns = out.columns.astype(str)
                return out.sort_index()
    lower = {str(c).lower(): c for c in df.columns}
    if field.lower() in lower:
        fcol = lower[field.lower()]
        tcol = next((lower[k] for k in ["datetime","timestamp","date","time"] if k in lower), None)
        scol = next((lower[k] for k in ["ticker","symbol","asset"] if k in lower), None)
        if tcol is not None and scol is not None:
            tmp = df[[tcol, scol, fcol]].copy()
            tmp[tcol] = pd.to_datetime(tmp[tcol])
            out = tmp.pivot(index=tcol, columns=scol, values=fcol)
            out.columns = out.columns.astype(str)
            return out.sort_index()
    if isinstance(df.index, pd.MultiIndex) and field in df.columns:
        tmp = df.reset_index()
        idx_cols = list(tmp.columns[:df.index.nlevels])
        tcol = next((c for c in idx_cols if pd.api.types.is_datetime64_any_dtype(tmp[c]) or "time" in str(c).lower() or "date" in str(c).lower()), None)
        scol = next((c for c in idx_cols if c != tcol), None)
        if tcol is not None and scol is not None:
            tmp[tcol] = pd.to_datetime(tmp[tcol], errors="coerce")
            out = tmp.pivot(index=tcol, columns=scol, values=field)
            out.columns = out.columns.astype(str)
            return out.sort_index()
    raise ValueError(f"Cannot extract {field} from DataFrame")


# ---------- 因子计算（完全原版）----------
def zprice(close, n):
    roll = close.rolling(n, min_periods=n)
    ma = roll.mean()
    sd = roll.std()
    sd = sd.where(sd != 0, np.nan)
    return ((close - ma) / sd).replace([np.inf, -np.inf], np.nan)


def volume_adjusted_return(close, volume, n):
    ret = close.pct_change(fill_method=None)
    vol_ma = volume.rolling(n, min_periods=max(2, n//2)).mean()
    adj = ret * (volume / vol_ma)
    return adj.rolling(max(2, min(12, n//2)), min_periods=2).sum().replace([np.inf, -np.inf], np.nan)


def volume_weighted_momentum(close, volume, n):
    return ((close / close.shift(n) - 1.0) * (volume / volume.shift(n))).replace([np.inf, -np.inf], np.nan)


def rsi_signal(close, n=12):
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-diff.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return (50 - rsi).replace([np.inf, -np.inf], np.nan)


def vwap_dev(close, volume, n=39):
    vwap = (close * volume).rolling(n, min_periods=max(2, n//2)).sum() / volume.rolling(n, min_periods=max(2, n//2)).sum()
    return (-(close - vwap) / vwap).replace([np.inf, -np.inf], np.nan)


def compute_raw_factors(close, volume, names):
    """批量计算原始因子（原逻辑）"""
    out = {}
    for name in names:
        if name.startswith("zprice_"):
            out[name] = zprice(close, int(name.split("_")[-1]))
        elif name.startswith("volume_adjusted_return_"):
            out[name] = volume_adjusted_return(close, volume, int(name.split("_")[-1]))
        elif name.startswith("volume_weighted_momentum_"):
            out[name] = volume_weighted_momentum(close, volume, int(name.split("_")[-1]))
        elif name == "rsi_signal_12":
            out[name] = rsi_signal(close, 12)
        elif name == "vwap_dev_39":
            out[name] = vwap_dev(close, volume, 39)
    return out


# ---------- 截面 rank 标准化（原版逐行）----------
def rank_cs(s):
    """原版 _rank_cs"""
    valid = s.astype("float64").dropna()
    out = pd.Series(np.nan, index=s.index, dtype="float32")
    if len(valid) < 2:
        return out
    ranks = valid.rank(method="average")
    n = float(len(valid))
    mu = (n + 1) / 2.0
    sd = np.sqrt((n * n - 1) / 12.0)
    z = (ranks - mu) / sd if sd > 0 else ranks - mu
    out.loc[valid.index] = z.astype("float32")
    return out


def rank_panel(df):
    """原版 _rank_panel：逐行调用 rank_cs"""
    rows = []
    for i in df.index:
        rows.append(rank_cs(df.loc[i]))
    return pd.DataFrame(rows, index=df.index, columns=df.columns, dtype="float32")
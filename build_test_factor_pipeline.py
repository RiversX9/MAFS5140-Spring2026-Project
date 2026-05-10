from pathlib import Path
import gc

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


HORIZONS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
VAR_PERIODS = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144]
EXPECTED_FACTORS = [
    "zprice_6",
    "zprice_12",
    "zprice_18",
    "zprice_24",
    "zprice_30",
    "zprice_42",
    "zprice_36",
    "volume_adjusted_return_12",
    "volume_adjusted_return_24",
    "zprice_48",
    "volume_adjusted_return_36",
    "volume_adjusted_return_48",
    "volume_adjusted_return_60",
    "volume_adjusted_return_72",
    "volume_adjusted_return_84",
    "volume_adjusted_return_96",
    "volume_adjusted_return_108",
    "volume_adjusted_return_120",
    "volume_adjusted_return_132",
    "volume_adjusted_return_144",
    "zprice_54",
    "volume_weighted_momentum_12",
    "rsi_signal_12",
    "zprice_60",
    "vwap_dev_39",
]


def create_overnight_mask(price_df: pd.DataFrame) -> pd.DataFrame:
    first_minute_index = price_df.groupby(price_df.index.date).head(1).index
    overnight_mask = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
    overnight_mask.loc[first_minute_index] = True
    return overnight_mask


def load_clean_close_from_panel(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    close_df = df.xs("close", axis=1, level="field").ffill()

    overnight_mask = create_overnight_mask(close_df)
    close_df = close_df.copy()
    close_df[overnight_mask] = np.nan
    close_df = close_df.sort_index()
    return close_df.astype("float32")


def load_clean_volume_from_panel(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    volume_df = df.xs("volume", axis=1, level="field").ffill().sort_index()
    return volume_df.astype("float32")


def vwap_deviation(close_df: pd.DataFrame, volume_df: pd.DataFrame, period: int = 39) -> pd.DataFrame:
    cum_vol = volume_df.rolling(period, min_periods=1).sum()
    cum_pv = (close_df * volume_df).rolling(period, min_periods=1).sum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    return (-(close_df - vwap) / vwap).replace([np.inf, -np.inf], np.nan).astype("float32")


def vol_weighted_momentum(close_df: pd.DataFrame, volume_df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    vol_prev = volume_df.shift(period)
    vol_ok = vol_prev.notna() & (vol_prev > 0) & volume_df.notna()
    vol_weight = (volume_df / vol_prev).where(vol_ok)

    close_prev = close_df.shift(period)
    ret_ok = close_prev.notna() & (close_prev > 0) & close_df.notna()
    ret = ((close_df - close_prev) / close_prev).where(ret_ok)
    return (ret * vol_weight).replace([np.inf, -np.inf], np.nan).astype("float32")


def volume_adjusted_return(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    vol_lookback: int = 12,
    ret_lookback: int = 6,
) -> pd.DataFrame:
    ret = close_df.pct_change(fill_method=None)
    vol_ratio = volume_df / volume_df.rolling(vol_lookback, min_periods=max(1, vol_lookback // 2)).mean()
    adj_ret = ret * vol_ratio
    adj_mom = adj_ret.rolling(ret_lookback, min_periods=max(1, ret_lookback // 2)).sum()
    return adj_mom.replace([np.inf, -np.inf], np.nan).astype("float32")


def rsi_signal(close_df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    delta = close_df.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=max(1, period // 2)).mean()
    avg_loss = loss.rolling(period, min_periods=max(1, period // 2)).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return ((30 - rsi) / 30).replace([np.inf, -np.inf], np.nan).astype("float32")


def build_raw_factors(close_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    factors: dict[str, pd.DataFrame] = {}
    for h in HORIZONS:
        f = np.log(close_df / close_df.shift(h))
        f = f.replace([np.inf, -np.inf], np.nan).astype("float32")
        factors[f"zprice_{h}"] = f
    return factors


def build_all_factors(close_df: pd.DataFrame, volume_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    factors = build_raw_factors(close_df)
    for p in VAR_PERIODS:
        factors[f"volume_adjusted_return_{p}"] = volume_adjusted_return(
            close_df, volume_df, vol_lookback=p, ret_lookback=p
        )
    factors["volume_weighted_momentum_12"] = vol_weighted_momentum(close_df, volume_df, period=12)
    factors["rsi_signal_12"] = rsi_signal(close_df, period=12)
    factors["vwap_dev_39"] = vwap_deviation(close_df, volume_df, period=39)
    return factors


def normalize_2_chunk(group: pd.DataFrame) -> pd.DataFrame:
    arr = group.to_numpy(dtype=np.float32, copy=True)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.inf

    n_rows, n_cols = arr.shape
    if n_cols <= 1:
        out = group.astype("float32").copy()
        out[nan_mask] = np.nan
        return out

    order = np.argsort(arr, axis=1, kind="mergesort")
    ranks = np.empty_like(arr, dtype=np.float32)
    row_ids = np.arange(n_rows)[:, None]
    ranks[row_ids, order] = np.arange(1, n_cols + 1, dtype=np.float32)

    mean = np.float32((n_cols + 1) / 2.0)
    std = np.float32(np.sqrt((n_cols**2 - 1) / 12.0))
    z = (ranks - mean) / std
    z[nan_mask] = np.nan
    return pd.DataFrame(z, index=group.index, columns=group.columns, dtype="float32")


def iter_daily_chunks(df: pd.DataFrame):
    if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
        grouped = df.groupby(level="date", sort=False)
    else:
        date_key = pd.Index(df.index.normalize(), name="date")
        grouped = df.groupby(date_key, sort=False)
    for _, chunk in grouped:
        yield chunk


def normalize_by_daily_chunks(df: pd.DataFrame, method: str = "rank") -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in iter_daily_chunks(df):
        if method == "rank":
            norm_chunk = normalize_2_chunk(chunk)
        else:
            raise ValueError("method must be 'rank'")
        chunks.append(norm_chunk.astype("float32"))
    if not chunks:
        return pd.DataFrame(index=df.index, columns=df.columns, dtype="float32")
    return pd.concat(chunks, axis=0).astype("float32")


def main() -> None:
    base = Path(__file__).resolve().parent
    train_path = base / "data_downloader" / "train.parquet"
    test_path = base / "data_downloader" / "test.parquet"

    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    out_raw_train = base / "factor_store" / "raw_train"
    out_raw_test = base / "factor_store" / "raw_test"
    out_std_test = base / "factor_store" / "std_test_from_train"
    out_std_train = base / "factor_store" / "std_train"
    out_stats = base / "factor_store" / "train_stats"
    for p in [
        out_raw_train,
        out_raw_test,
        out_std_test,
        out_std_train,
        out_stats,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    # 清除旧文件，确保可重复性
    for p in [
        out_raw_train,
        out_raw_test,
        out_std_test,
        out_std_train,
        out_stats,
    ]:
        for fp in p.glob("*.parquet"):
            fp.unlink()

    print("Loading and cleaning train/test close+volume...")
    close_train = load_clean_close_from_panel(train_path)
    close_test = load_clean_close_from_panel(test_path)
    volume_train = load_clean_volume_from_panel(train_path)
    volume_test = load_clean_volume_from_panel(test_path)

    common_cols = (
        close_train.columns.intersection(close_test.columns)
        .intersection(volume_train.columns)
        .intersection(volume_test.columns)
    )
    close_train = close_train[common_cols]
    close_test = close_test[common_cols]
    volume_train = volume_train[common_cols]
    volume_test = volume_test[common_cols]

    print("Building full raw factors...")
    raw_train = build_all_factors(close_train, volume_train)
    raw_test = build_all_factors(close_test, volume_test)

    missing_train = sorted(set(EXPECTED_FACTORS) - set(raw_train.keys()))
    missing_test = sorted(set(EXPECTED_FACTORS) - set(raw_test.keys()))
    if missing_train or missing_test:
        raise RuntimeError(
            f"Missing factors. train_missing={missing_train}, test_missing={missing_test}"
        )

    print("Saving raw factors + train-fitted standardization...")
    for name in tqdm(EXPECTED_FACTORS, desc="factor pipeline"):
        train_df = raw_train[name]
        test_df = raw_test[name]

        train_df.to_parquet(out_raw_train / f"{name}.parquet", compression="zstd")
        test_df.to_parquet(out_raw_test / f"{name}.parquet", compression="zstd")

        std_train = normalize_by_daily_chunks(train_df, method="rank")
        std_test = normalize_by_daily_chunks(test_df, method="rank")

        # 保留占位符文件（虽然未使用均值/标准差）
        pd.Series(index=train_df.columns, data=np.nan, dtype="float32").to_frame("mean").to_parquet(
            out_stats / f"{name}_mean.parquet", compression="zstd"
        )
        pd.Series(index=train_df.columns, data=np.nan, dtype="float32").to_frame("std").to_parquet(
            out_stats / f"{name}_std.parquet", compression="zstd"
        )

        std_train.to_parquet(out_std_train / f"{name}.parquet", compression="zstd")
        std_test.to_parquet(out_std_test / f"{name}.parquet", compression="zstd")
        del train_df, test_df, std_train, std_test
        gc.collect()

    print("Done.")
    print(f"- raw_train: {out_raw_train}")
    print(f"- raw_test: {out_raw_test}")
    print(f"- std_train: {out_std_train}")
    print(f"- std_test_from_train: {out_std_test}")
    print(f"- saved factor count: {len(EXPECTED_FACTORS)}")


if __name__ == "__main__":
    main()
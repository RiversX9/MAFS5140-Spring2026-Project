# strategy.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import subprocess
import sys
import json

from common import compute_raw_factors, rank_cs


# ======================== William 策略（内部类） ========================
class _WilliamStrategy:
    """William 策略：纯计数器版本，无前视偏差，在线 PCA 自适应权重"""

    def __init__(self):
        self.refit_bars = 1
        self._window_bars = 390
        self.threshold = 0.9
        self.top_pct = 0.05

        self._bars_per_day = 78
        self._refit_bars = max(1, int(self.refit_bars))
        self._bar_count = 0
        self._test_step = 0
        self._test_len = 0
        self._cached_factor_weights = None

        self._factor_names = [
            "zprice_6", "zprice_12", "zprice_18", "zprice_24", "zprice_30",
            "zprice_42", "zprice_36", "volume_adjusted_return_12",
            "volume_adjusted_return_24", "zprice_48", "volume_adjusted_return_36",
            "volume_adjusted_return_48", "volume_adjusted_return_60",
            "volume_adjusted_return_72", "volume_adjusted_return_84",
            "volume_adjusted_return_96", "volume_adjusted_return_108",
            "volume_adjusted_return_120", "volume_adjusted_return_132",
            "volume_adjusted_return_144", "zprice_54",
            "volume_weighted_momentum_12", "rsi_signal_12",
            "zprice_60", "vwap_dev_39",
        ]

        self._train_factors = {}
        self._test_factors = {}
        self._active_factors = []
        self._history = {}

        self._load_factor_panels()
        self._init_train_history()
        self._build_test_index_map()

    # ---- 以下方法与 William 原始实现完全一致 ----
    def _build_test_index_map(self):
        if not self._active_factors:
            self._test_len = 0
            return
        self._test_len = len(self._test_factors[self._active_factors[0]])

    def _candidate_data_roots(self):
        base = Path(__file__).resolve().parent
        return [base, base / "mini_2" / "mini_2"]

    def _ensure_factor_pipeline_outputs(self):
        base = Path(__file__).resolve().parent
        out_std_train = base / "factor_store" / "std_train"
        out_std_test = base / "factor_store" / "std_test_from_train"

        def _has_complete_outputs():
            if not (out_std_train.exists() and out_std_test.exists()):
                return False
            train_files = sorted(out_std_train.glob("*.parquet"))
            test_files = sorted(out_std_test.glob("*.parquet"))
            if len(train_files) == 0 or len(test_files) == 0:
                return False
            train_names = {p.name for p in train_files}
            test_names = {p.name for p in test_files}
            return len(train_names.intersection(test_names)) > 0

        if _has_complete_outputs():
            return

        pipeline_script = base / "build_test_factor_pipeline.py"
        if not pipeline_script.exists():
            raise FileNotFoundError(f"Missing pipeline script: {pipeline_script}")

        subprocess.run([sys.executable, str(pipeline_script)], cwd=str(base), check=True)

        if not _has_complete_outputs():
            raise RuntimeError("Pipeline outputs still incomplete.")

    def _load_factor_panels(self):
        self._ensure_factor_pipeline_outputs()
        train_dir = None
        test_dir = None
        root_used = None
        for root in self._candidate_data_roots():
            td = root / "factor_store" / "std_train"
            vd = root / "factor_store" / "std_test_from_train"
            if td.exists() and vd.exists():
                train_dir = td
                test_dir = vd
                root_used = root
                break
        if train_dir is None or test_dir is None:
            raise RuntimeError("No factor store found.")

        raw_train = {}
        raw_test = {}
        for name in self._factor_names:
            p_tr = train_dir / f"{name}.parquet"
            p_te = test_dir / f"{name}.parquet"
            if not p_tr.exists() or not p_te.exists():
                continue
            df_tr = pd.read_parquet(p_tr).astype("float32")
            df_te = pd.read_parquet(p_te).astype("float32")
            common_cols = df_tr.columns.intersection(df_te.columns)
            if len(common_cols) == 0:
                continue
            raw_train[name] = df_tr.loc[:, common_cols]
            raw_test[name] = df_te.loc[:, common_cols]

        if not raw_train:
            raise RuntimeError("No raw standardized factors loaded.")

        merged_train_dir, merged_test_dir = self._materialize_merged_factors(
            root=root_used, raw_train=raw_train, raw_test=raw_test
        )

        self._train_factors = {}
        self._test_factors = {}
        self._active_factors = []
        for p_tr in sorted(merged_train_dir.glob("*.parquet")):
            name = p_tr.stem
            p_te = merged_test_dir / p_tr.name
            if not p_te.exists():
                continue
            df_tr = pd.read_parquet(p_tr).astype("float32")
            df_te = pd.read_parquet(p_te).astype("float32")
            common_cols = df_tr.columns.intersection(df_te.columns)
            if len(common_cols) == 0:
                continue
            self._train_factors[name] = df_tr.loc[:, common_cols]
            self._test_factors[name] = df_te.loc[:, common_cols]
            self._active_factors.append(name)

        for name in self._active_factors:
            self._history[name] = []

    def _fit_merge_group_weight_map(self, raw_train):
        from numpy.linalg import eigh
        factor_series = []
        names = []
        for name, df in raw_train.items():
            s = df.median(axis=1, skipna=True).astype("float32")
            factor_series.append(s.rename(name))
            names.append(name)
        x = pd.concat(factor_series, axis=1).dropna(how="any")
        if x.empty:
            raise RuntimeError("Empty train signatures.")
        std_mask = x.std(axis=0, skipna=True) > 1e-8
        x = x.loc[:, std_mask]
        if x.shape[1] == 0:
            raise RuntimeError("All factors near-constant.")

        corr = pd.DataFrame(
            self._safe_corrcoef(x.to_numpy(dtype=np.float32, copy=False)),
            index=x.columns,
            columns=x.columns,
        )
        groups_idx = self._build_groups(corr)
        valid_names = list(corr.index)

        group_weight_map = {}
        gid = 0
        x_np = x.to_numpy(dtype=np.float32, copy=False)
        for g in groups_idx:
            group_names = [valid_names[i] for i in g]
            merged_name = f"merged_{gid}"
            if len(group_names) == 1:
                group_weight_map[merged_name] = {group_names[0]: 1.0}
                gid += 1
                continue

            xg = x_np[:, g]
            if xg.ndim != 2 or xg.shape[0] < 2 or xg.shape[1] < 2:
                eq = 1.0 / len(group_names)
                group_weight_map[merged_name] = {n: float(eq) for n in group_names}
                gid += 1
                continue
            xg = xg - np.nanmean(xg, axis=0, keepdims=True)
            cov = np.cov(xg, rowvar=False)
            eigvals, eigvecs = eigh(cov)
            v = eigvecs[:, np.argmax(eigvals)]
            sign = np.sign(np.nanmean(xg @ v))
            if sign == 0:
                sign = 1.0
            v = v * sign
            denom = np.sum(np.abs(v))
            if denom > 0:
                v = v / denom
            group_weight_map[merged_name] = {n: float(w) for n, w in zip(group_names, v)}
            gid += 1
        return group_weight_map

    @staticmethod
    def _merge_from_weight_map(factor_map, group_weight_map):
        out = {}
        for merged_name, comp in group_weight_map.items():
            items = [(n, float(w)) for n, w in comp.items() if n in factor_map]
            if not items:
                continue
            base_cols = None
            base_idx = None
            for n, _ in items:
                df = factor_map[n]
                base_cols = df.columns if base_cols is None else base_cols.intersection(df.columns)
                base_idx = df.index if base_idx is None else base_idx.intersection(df.index)
            if base_cols is None or base_idx is None or len(base_cols) == 0 or len(base_idx) == 0:
                continue

            acc = None
            for n, w in items:
                arr = factor_map[n].loc[base_idx, base_cols].to_numpy(dtype=np.float32, copy=False)
                if acc is None:
                    acc = np.zeros_like(arr, dtype=np.float32)
                acc += arr * np.float32(w)
            out[merged_name] = pd.DataFrame(acc, index=base_idx, columns=base_cols, dtype="float32")
        return out

    def _materialize_merged_factors(self, root, raw_train, raw_test):
        from numpy.linalg import eigh
        merged_train_dir = root / "factor_store" / "merged_train"
        merged_test_dir = root / "factor_store" / "merged_test_from_train"
        merged_meta_dir = root / "factor_store" / "merged_meta"
        merged_train_dir.mkdir(parents=True, exist_ok=True)
        merged_test_dir.mkdir(parents=True, exist_ok=True)
        merged_meta_dir.mkdir(parents=True, exist_ok=True)

        group_weight_map = self._fit_merge_group_weight_map(raw_train)
        merged_train = self._merge_from_weight_map(raw_train, group_weight_map)
        merged_test = self._merge_from_weight_map(raw_test, group_weight_map)
        if not merged_train or not merged_test:
            raise RuntimeError("Merged factor generation failed.")

        for fp in merged_train_dir.glob("*.parquet"):
            fp.unlink()
        for fp in merged_test_dir.glob("*.parquet"):
            fp.unlink()

        common_merged = sorted(set(merged_train.keys()).intersection(merged_test.keys()))
        for name in common_merged:
            merged_train[name].to_parquet(merged_train_dir / f"{name}.parquet", compression="zstd")
            merged_test[name].to_parquet(merged_test_dir / f"{name}.parquet", compression="zstd")

        json_path = merged_meta_dir / "group_weight_map.json"
        json_path.write_text(json.dumps({
            "threshold": float(self.threshold),
            "factor_names": sorted(raw_train.keys()),
            "merged_names": common_merged,
            "group_weight_map": group_weight_map,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        return merged_train_dir, merged_test_dir

    def _init_train_history(self):
        if not self._active_factors:
            return
        idx_train = self._train_factors[self._active_factors[0]].index
        for ts in idx_train[-self._window_bars:]:
            for name in self._active_factors:
                self._history[name].append(self._train_factors[name].loc[ts])

    @staticmethod
    def _safe_corrcoef(x):
        c = np.corrcoef(x, rowvar=False)
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(c, 1.0)
        return c

    def _build_groups(self, corr_df):
        k = corr_df.shape[0]
        mat = corr_df.to_numpy(dtype=float)
        visited = [False] * k
        groups = []
        idx = np.arange(k)
        for s in range(k):
            if visited[s]:
                continue
            stack = [s]
            visited[s] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                nbrs = np.where((np.abs(mat[u]) >= self.threshold) & (idx != u))[0]
                for v in nbrs:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            groups.append(sorted(comp))
        return groups

    def _fit_factor_weights(self):
        if not self._active_factors:
            return np.array([], dtype=float)

        cols = []
        n_obs = len(self._history[self._active_factors[0]])
        for name in self._active_factors:
            agg = np.full(n_obs, np.nan, dtype=np.float32)
            for i, s in enumerate(self._history[name]):
                if s is None:
                    continue
                arr = s.to_numpy(dtype=np.float32, copy=False)
                if arr.size == 0:
                    continue
                valid = np.isfinite(arr)
                if not np.any(valid):
                    continue
                agg[i] = np.median(arr[valid])
            cols.append(pd.Series(agg, name=name))

        x = pd.concat(cols, axis=1).dropna()
        if x.empty or len(x) < 10:
            return np.ones(len(self._active_factors), dtype=float) / max(1, len(self._active_factors))

        std_mask = x.std(axis=0, skipna=True) > 1e-8
        valid_names = list(x.columns[std_mask])
        if len(valid_names) == 0:
            return np.ones(len(self._active_factors), dtype=float) / max(1, len(self._active_factors))
        x = x[valid_names]

        corr = pd.DataFrame(
            self._safe_corrcoef(x.to_numpy(dtype=np.float32, copy=False)),
            index=valid_names,
            columns=valid_names,
        )
        groups = self._build_groups(corr)

        w_map = {n: 0.0 for n in self._active_factors}
        X = x.to_numpy(dtype=float, copy=False)
        for g in groups:
            if len(g) == 1:
                w_map[valid_names[g[0]]] = 1.0
                continue
            Xg = X[:, g]
            Xg = Xg - np.nanmean(Xg, axis=0, keepdims=True)
            cov = np.cov(Xg, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            v = eigvecs[:, np.argmax(eigvals)]
            sign = np.sign(np.nanmean(Xg @ v))
            if sign == 0:
                sign = 1.0
            v = v * sign
            denom = np.sum(np.abs(v))
            if denom > 0:
                v = v / denom
            for j, gi in enumerate(g):
                w_map[valid_names[gi]] = float(v[j])

        w = np.array([w_map[n] for n in self._active_factors], dtype=float)
        norm = np.sum(np.abs(w))
        if norm > 0:
            w = w / norm
        else:
            w = np.ones(len(w), dtype=float) / max(1, len(w))
        return w

    @staticmethod
    def _long_only_top(signal, top_pct):
        s = signal.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty or len(s) < 10:
            return pd.Series(0.0, index=signal.index, dtype=float)
        n = max(1, int(np.ceil(len(s) * float(top_pct))))
        top = s.nlargest(n)
        w = pd.Series(0.0, index=signal.index, dtype=float)
        if len(top) > 0:
            w.loc[top.index] = 1.0 / len(top)
        return w

    @staticmethod
    def _extract_tickers(data):
        if isinstance(data, pd.DataFrame):
            return list(data.index)
        if isinstance(data, dict):
            return list(data.keys())
        return []

    def _get_test_factor_snapshot(self, tickers):
        if not self._active_factors:
            return pd.DataFrame(index=tickers)
        if self._test_step >= self._test_len:
            return pd.DataFrame(index=tickers)

        rows = []
        for name in self._active_factors:
            row = self._test_factors[name].iloc[self._test_step].reindex(tickers)
            rows.append(row.rename(name))
        self._test_step += 1
        return pd.concat(rows, axis=1)

    def step(self, current_market_data):
        tickers = self._extract_tickers(current_market_data)
        self._bar_count += 1
        if len(tickers) == 0:
            return pd.Series(dtype="float64")

        fac_now = self._get_test_factor_snapshot(tickers)
        if fac_now.empty:
            return pd.Series(0.0, index=tickers, dtype="float64")

        for name in self._active_factors:
            self._history[name].append(fac_now[name])
            if len(self._history[name]) > self._window_bars:
                self._history[name] = self._history[name][-self._window_bars:]

        if self._cached_factor_weights is None:
            self._cached_factor_weights = np.ones(len(self._active_factors), dtype=float) / len(self._active_factors)
        w_t = pd.Series(self._cached_factor_weights, index=self._active_factors, dtype="float32")

        weighted_names = [n for n in self._active_factors if n in fac_now.columns and n in w_t.index]
        if len(weighted_names) == 0:
            return pd.Series(0.0, index=tickers, dtype="float64")

        sig = pd.Series(0.0, index=tickers, dtype=float)
        for name in weighted_names:
            wn = float(w_t[name]) if pd.notna(w_t[name]) else 0.0
            if wn == 0.0:
                continue
            sig = sig.add(fac_now[name].astype(float) * wn, fill_value=0.0)

        abs_sum = float(np.nansum(np.abs(w_t[weighted_names].to_numpy(dtype=float))))
        if abs_sum > 0:
            sig = sig / abs_sum
        sig = sig.replace([np.inf, -np.inf], np.nan)

        enough_hist = len(self._history[self._active_factors[0]]) >= max(100, self._bars_per_day)
        should_refit = False
        if enough_hist:
            if self._refit_bars <= 1:
                should_refit = True
            elif (self._bar_count % self._refit_bars) == 0:
                should_refit = True
        if should_refit:
            self._cached_factor_weights = self._fit_factor_weights()

        w = self._long_only_top(sig, self.top_pct).reindex(tickers).fillna(0.0)
        total = float(w.sum())
        if total > 0:
            w = w / total
        return w.reindex(tickers).fillna(0.0).astype("float64")


# ======================== 固定共识策略（内部类） ========================
class _FixedConsensusStrategy:
    """
    Fixed consensus strategy based on four rolling-window training results.

    This strategy does NOT train:
    - no train parquet is read
    - no IC / IR fitting is run
    - no LASSO is run
    - no LightGBM is run
    - no validation parameter optimization is run

    The selected factors are consensus factors supported across the four
    rolling-window fixed strategies.

    Fixed settings:
    - top_pct = 0.05
    - rebalance_every = 1
    - selected factors = 10
    - IC-sign-adjusted equal weights
    """

    MODE = "fixed_consensus_4rolling_method4_selected10_top5"

    def __init__(self):
        # ===== Consensus portfolio parameters =====
        self.top_pct = 0.05
        self.rebalance_every = 1
        self.bars_per_day = 78

        # ===== Consensus selected factors =====
        # Selected from four rolling windows by repeated appearance and stable IC direction.
        self.selected_factors = [
            "volume_adjusted_return_12",  # appeared in all 4 windows
            "zprice_6",                   # appeared in 3/4 windows
            "zprice_9",                   # appeared in 3/4 windows
            "log_mom_9",                  # appeared in 3/4 windows
            "volume_adjusted_return_6",   # appeared in 3/4 windows
            "volume_adjusted_return_18",  # appeared in 3/4 windows
            "log_mom_6",                  # appeared in recent rolling windows
            "vwap_dev_24",                # repeated VWAP mean-reversion factor
            "volume_zscore_12",           # repeated in recent rolling windows
            "zprice_15",                  # repeated price-reversal factor
        ]

        # IC-direction-adjusted equal weights.
        # Negative weights mean the factor is used in reverse direction.
        # Positive weights mean the factor is used in original direction.
        self.factor_weights = {
            "volume_adjusted_return_12": -0.1,
            "zprice_6": -0.1,
            "zprice_9": -0.1,
            "log_mom_9": -0.1,
            "volume_adjusted_return_6": -0.1,
            "volume_adjusted_return_18": -0.1,
            "log_mom_6": -0.1,
            "vwap_dev_24": 0.1,
            "volume_zscore_12": 0.1,
            "zprice_15": -0.1,
        }

        self.factor_signs = {k: np.sign(v) for k, v in self.factor_weights.items()}

        # ===== Runtime state =====
        self.history_close = None
        self.history_volume = None
        self.bar_count = 0
        self.last_weights = None

        self.max_window = self._infer_max_window(self.selected_factors)
        self.min_history = max(40, self.max_window + 5)
        self.max_history = max(260, self.max_window + 40)

        print("\n========== CONSENSUS FIXED STRATEGY INIT ==========")
        print("MODE:", self.MODE)
        print("No training will be performed.")
        print("top_pct:", self.top_pct)
        print("rebalance_every:", self.rebalance_every)
        print("selected_factors:", self.selected_factors)
        print("factor_weights:", self.factor_weights)
        print("min_history:", self.min_history)
        print("max_history:", self.max_history)
        print("==================================================\n")

    # ============================================================
    # Utility
    # ============================================================
    def _infer_max_window(self, factor_names):
        max_w = 12
        for name in factor_names:
            for token in name.split("_"):
                if token.isdigit():
                    max_w = max(max_w, int(token))
        return max_w

    def _rank_standardize_cross_section(self, s: pd.Series) -> pd.Series:
        """
        Cross-sectional rank standardization.

        For each bar:
        - rank stocks by raw factor value
        - transform rank to approximately zero mean and unit variance
        """
        valid = s.astype("float64").dropna()
        out = pd.Series(np.nan, index=s.index, dtype="float32")

        if len(valid) < 2:
            return out

        ranks = valid.rank(method="average")
        n = float(len(valid))
        rank_mean = (n + 1.0) / 2.0
        rank_std = np.sqrt((n ** 2 - 1.0) / 12.0)

        if rank_std > 0:
            z = (ranks - rank_mean) / rank_std
        else:
            z = ranks - rank_mean

        out.loc[valid.index] = z.astype("float32")
        return out

    # ============================================================
    # Factor definitions
    # ============================================================
    def _zprice(self, close: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Price z-score:
        (current price - rolling mean) / rolling std
        """
        roll = close.rolling(n, min_periods=max(3, n // 2))
        ma = roll.mean()
        sd = roll.std().replace(0, np.nan)
        return ((close - ma) / sd).replace([np.inf, -np.inf], np.nan).astype("float32", copy=False)

    def _log_mom(self, close: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Log momentum:
        log(P_t / P_{t-n})
        """
        return np.log(close / close.shift(n)).replace([np.inf, -np.inf], np.nan).astype("float32", copy=False)

    def _volume_adjusted_return(self, close: pd.DataFrame, volume: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Volume-adjusted return:
        short-horizon return weighted by relative volume.
        """
        ret = close.pct_change(fill_method=None)
        vol_ma = volume.rolling(n, min_periods=max(2, n // 2)).mean()
        vol_ratio = volume / vol_ma
        adj_ret = ret * vol_ratio
        ret_lookback = max(2, min(12, n // 2))
        out = adj_ret.rolling(ret_lookback, min_periods=2).sum()
        return out.replace([np.inf, -np.inf], np.nan).astype("float32", copy=False)

    def _vwap_dev(self, close: pd.DataFrame, volume: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        VWAP deviation signal:
        positive when price is below rolling VWAP.
        """
        pv = close * volume
        vwap = (
            pv.rolling(n, min_periods=max(2, n // 2)).sum()
            / volume.rolling(n, min_periods=max(2, n // 2)).sum()
        )
        return (-(close - vwap) / vwap).replace([np.inf, -np.inf], np.nan).astype("float32", copy=False)

    def _volume_zscore(self, volume: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Volume z-score:
        (current volume - rolling mean volume) / rolling std volume
        """
        ma = volume.rolling(n, min_periods=max(2, n // 2)).mean()
        sd = volume.rolling(n, min_periods=max(2, n // 2)).std().replace(0, np.nan)
        return ((volume - ma) / sd).replace([np.inf, -np.inf], np.nan).astype("float32", copy=False)

    def _compute_one_raw_factor(self, close: pd.DataFrame, volume: pd.DataFrame, name: str) -> pd.DataFrame:
        if name.startswith("zprice_"):
            return self._zprice(close, int(name.split("_")[-1]))

        if name.startswith("log_mom_"):
            return self._log_mom(close, int(name.split("_")[-1]))

        if name.startswith("volume_adjusted_return_"):
            return self._volume_adjusted_return(close, volume, int(name.split("_")[-1]))

        if name.startswith("vwap_dev_"):
            return self._vwap_dev(close, volume, int(name.split("_")[-1]))

        if name.startswith("volume_zscore_"):
            return self._volume_zscore(volume, int(name.split("_")[-1]))

        raise ValueError(f"Unknown factor: {name}")

    # ============================================================
    # Online backtest step
    # ============================================================
    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        tickers = current_market_data.index

        if "close" not in current_market_data.columns or "volume" not in current_market_data.columns:
            return pd.Series(0.0, index=tickers, dtype="float32")

        # Rebalance rule.
        if self.last_weights is not None and self.bar_count % self.rebalance_every != 0:
            self.bar_count += 1
            return self.last_weights.reindex(tickers).fillna(0.0).astype("float32")

        current_close = current_market_data["close"].astype("float32")
        current_volume = current_market_data["volume"].astype("float32")

        next_idx = 0 if self.history_close is None else len(self.history_close)

        close_row = pd.DataFrame(
            [current_close.values],
            index=[next_idx],
            columns=tickers,
            dtype="float32",
        )
        volume_row = pd.DataFrame(
            [current_volume.values],
            index=[next_idx],
            columns=tickers,
            dtype="float32",
        )

        # Approximate overnight removal:
        # set first bar of each day to NaN to reduce overnight jump effects.
        if self.bar_count % self.bars_per_day == 0:
            close_row.iloc[0, :] = np.nan
            volume_row.iloc[0, :] = np.nan

        if self.history_close is None:
            self.history_close = close_row
            self.history_volume = volume_row
            self.bar_count += 1
            weights = pd.Series(0.0, index=tickers, dtype="float32")
            self.last_weights = weights.copy()
            return weights

        self.history_close = pd.concat([self.history_close, close_row])
        self.history_volume = pd.concat([self.history_volume, volume_row])

        if len(self.history_close) > self.max_history:
            self.history_close = self.history_close.iloc[-self.max_history:].copy()
            self.history_volume = self.history_volume.iloc[-self.max_history:].copy()
            self.history_close.index = range(len(self.history_close))
            self.history_volume.index = range(len(self.history_volume))

        if len(self.history_close) < self.min_history:
            self.bar_count += 1
            weights = pd.Series(0.0, index=tickers, dtype="float32")
            self.last_weights = weights.copy()
            return weights

        latest_idx = self.history_close.index[-1]

        signal = pd.Series(0.0, index=tickers, dtype="float32")
        used = 0

        for name in self.selected_factors:
            try:
                raw_df = self._compute_one_raw_factor(self.history_close, self.history_volume, name)
                raw_series = raw_df.loc[latest_idx]
            except Exception:
                continue

            if raw_series.isna().all():
                continue

            z = self._rank_standardize_cross_section(raw_series).reindex(tickers)
            weight = self.factor_weights.get(name, 0.0)

            signal = signal.add(weight * z, fill_value=0.0)
            used += 1

        self.bar_count += 1

        if used == 0:
            weights = pd.Series(0.0, index=tickers, dtype="float32")
            self.last_weights = weights.copy()
            return weights

        valid = signal.replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 10:
            weights = pd.Series(0.0, index=tickers, dtype="float32")
            self.last_weights = weights.copy()
            return weights

        n_long = max(1, int(np.ceil(len(valid) * self.top_pct)))
        top = valid.nlargest(n_long).index

        weights = pd.Series(0.0, index=tickers, dtype="float32")
        weights.loc[top] = 1.0 / float(n_long)

        self.last_weights = weights.copy()
        return weights


# ======================== 集成策略（方案B：滚动夏普加权） ========================
class Strategy:
    """
    集成策略：动态加权
    采用方案B：基于滚动窗口内各子策略的夏普比率来分配权重，
    使用 Softmax 归一化，温度系数可调。
    """

    def __init__(self, lookback: int = 20, min_periods: int = 5, temperature: float = 2.0):
        self.william = _WilliamStrategy()
        self.fixed = _FixedConsensusStrategy()

        # Stacking 参数
        self.lookback = lookback            # 滚动窗口长度（调仓次数）
        self.min_periods = min_periods      # 最少需要多少期才启用动态权重
        self.temperature = temperature      # softmax 温度，越大权重差异越明显

        # 记录每个子策略的逐期收益率（每 bar 一次）
        self.william_returns = []
        self.fixed_returns = []

        # 上一期状态（用于计算当期收益）
        self.last_weights_w = None
        self.last_weights_f = None
        self.last_close = None

        self.step_count = 0

        print("\n========== DYNAMIC STACKING STRATEGY (SHARPE RATIO) ==========")
        print(f"lookback = {lookback}, min_periods = {min_periods}, temperature = {temperature}")
        print("Dynamic weights update: based on rolling Sharpe ratio of each strategy")
        print("==================================================================\n")

    def _rolling_sharpe(self, returns_list):
        """从 returns_list 中取最近 lookback 期，计算夏普比率（均值/标准差）"""
        if len(returns_list) < self.min_periods:
            return 0.0
        arr = np.array(returns_list[-self.lookback:])
        mean = arr.mean()
        std = arr.std()
        if std < 1e-8:
            return 0.0
        return mean / std

    def _dynamic_weights(self):
        """返回 (alpha, beta)，分别为 William 和 Fixed 的权重。"""
        sr_w = self._rolling_sharpe(self.william_returns)
        sr_f = self._rolling_sharpe(self.fixed_returns)

        # 如果历史不足，返回等权
        if len(self.william_returns) < self.min_periods or len(self.fixed_returns) < self.min_periods:
            return 0.5, 0.5

        # Softmax 归一化
        exp_w = np.exp(self.temperature * sr_w)
        exp_f = np.exp(self.temperature * sr_f)
        total = exp_w + exp_f
        if total <= 0:
            return 0.5, 0.5
        alpha = exp_w / total
        beta = exp_f / total
        return alpha, beta

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        tickers = current_market_data.index
        close = current_market_data["close"]

        # ------------------------------------------------------------------
        # 1. 如果有上一期持仓，计算上一期的组合收益并更新历史列表
        # ------------------------------------------------------------------
        if self.last_weights_w is not None and self.last_weights_f is not None and self.last_close is not None:
            # 计算本期（当前 bar）相对于上一期 close 的收益率
            ret = (close / self.last_close - 1.0).reindex(tickers, fill_value=0.0)

            # 子策略的收益贡献（持仓权重 × 个股收益）
            r_w = (self.last_weights_w * ret).sum()
            r_f = (self.last_weights_f * ret).sum()

            self.william_returns.append(r_w)
            self.fixed_returns.append(r_f)

        # ------------------------------------------------------------------
        # 2. 获取当前子策略权重
        # ------------------------------------------------------------------
        w_w = self.william.step(current_market_data)
        w_f = self.fixed.step(current_market_data)

        w_w = w_w.reindex(tickers, fill_value=0.0)
        w_f = w_f.reindex(tickers, fill_value=0.0)

        # ------------------------------------------------------------------
        # 3. 计算动态权重
        # ------------------------------------------------------------------
        alpha, beta = self._dynamic_weights()

        # ------------------------------------------------------------------
        # 4. 集成
        # ------------------------------------------------------------------
        combined = alpha * w_w + beta * w_f
        # 理论上 combined 的和应该为 1，但为防止浮点误差，显式归一化一次
        total_weight = combined.sum()
        if total_weight > 1e-6:
            combined = combined / total_weight
        else:
            # 极端情况：全为零，返回等权
            combined = pd.Series(1.0 / len(tickers), index=tickers, dtype="float32")

        # ------------------------------------------------------------------
        # 5. 保存状态供下一期使用
        # ------------------------------------------------------------------
        self.last_weights_w = w_w
        self.last_weights_f = w_f
        self.last_close = close
        self.step_count += 1

        return combined.astype("float32")
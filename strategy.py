"""
策略文件 - 基于 Round2.py 的 IC 加权多因子策略
适配框架：engine.py 调用 step(current_market_data)
"""

import pandas as pd
import numpy as np
from pathlib import Path


class Strategy:
    def __init__(self):
        # ---------- 因子池（与 Round2.py 一致）----------
        self.factor_names = [
            "zprice_12",
            "zprice_36",
            "vol_weighted_momentum_6",
            "volume_adjusted_return_12",
        ]

        # 历史数据缓存
        self.history_close = None
        self.history_ret = None
        self.history_volume = None          # 用于成交量相关因子
        self.min_history = 100

        # ---------- 加载训练期 IC 权重 ----------
        ic_path = Path("ic_weight.parquet")
        if not ic_path.exists():
            raise FileNotFoundError(
                "缺少 ic_weight.parquet，请先运行 Round2.py 生成该文件"
            )
        self.ic_weight = pd.read_parquet(ic_path).sort_index()
        # 确保只保留所需因子列
        missing = set(self.factor_names) - set(self.ic_weight.columns)
        if missing:
            raise ValueError(f"ic_weight.parquet 缺少因子列: {missing}")
        self.ic_weight = self.ic_weight[self.factor_names].astype("float32")

        # ---------- 加载训练期全局均值和标准差（用于新因子标准化）----------
        raw_dir = Path("factor_store/raw")
        self.factor_mean = {}
        self.factor_std = {}
        for name in self.factor_names:
            fp = raw_dir / f"{name}.parquet"
            if not fp.exists():
                raise FileNotFoundError(f"缺少训练期原始因子文件: {fp}")
            df = pd.read_parquet(fp)
            vals = df.values.ravel()
            vals = vals[np.isfinite(vals)]
            self.factor_mean[name] = float(np.mean(vals))
            self.factor_std[name] = float(np.std(vals))

        # 上一期的权重（用于换手率平滑，可选）
        self.prev_weights = None

    # ======================== 因子计算函数 ========================
    def _zprice_n(self, close_df, period=1):
        roll = close_df.rolling(period, min_periods=period)
        ma = roll.mean()
        std = roll.std()
        z = (close_df - ma) / std
        z = z.where(std != 0, np.nan)
        return z

    def _vol_weighted_momentum(self, ret_df, period=6):
        """
        波动率加权动量：动量 / 波动率，波动率用 period 期标准差。
        若波动率为 0，则设为 NaN。
        """
        momentum = ret_df.rolling(period, min_periods=period).sum()
        volatility = ret_df.rolling(period, min_periods=period).std()
        # 避免除零
        weighted = momentum / volatility.where(volatility != 0, np.nan)
        return weighted

    def _volume_adjusted_return(self, close_df, volume_df, period=12):
        """
        成交量调整收益：收益率 / 平均成交量（过去 period 期）。
        收益率用 close_df.pct_change()，为避免未来信息，shift(1) 后再除。
        """
        ret = close_df.pct_change().shift(1)  # t-1 期收益率，避免未来信息
        avg_vol = volume_df.rolling(period, min_periods=period).mean()
        adjusted = ret / avg_vol.where(avg_vol != 0, np.nan)
        return adjusted

    # ======================== 因子标准化（全局Z-score + 截面去极值） ========================
    def _standardize_factor(self, raw_series, mean, std):
        if raw_series.isna().all():
            return raw_series.astype("float32")
        z = (raw_series - mean) / std
        # 截面去极值（5倍标准差）
        cross_mean = float(z.mean())
        cross_std = float(z.std())
        if cross_std > 0:
            upper = cross_mean + 5 * cross_std
            lower = cross_mean - 5 * cross_std
            z = z.clip(lower=lower, upper=upper)
        return z.astype("float32")

    # ======================== 主步进函数 ========================
    def step(self, current_market_data):
        # 1. 提取数据
        if "close" not in current_market_data.columns:
            return pd.Series(0.0, index=current_market_data.index)
        current_prices = current_market_data["close"]
        tickers = current_prices.index

        # 成交量（若存在）
        if "volume" in current_market_data.columns:
            current_volume = current_market_data["volume"]
        else:
            # 若无成交量数据，则 volume_adjusted_return_12 因子将不可用，策略会自动跳过
            current_volume = pd.Series(np.nan, index=tickers)

        # 2. 更新时间戳（用整数索引避免时区问题）
        if self.history_close is None:
            next_idx = 0
        else:
            next_idx = len(self.history_close)

        current_close_df = pd.DataFrame([current_prices.values], index=[next_idx], columns=tickers)
        current_volume_df = pd.DataFrame([current_volume.values], index=[next_idx], columns=tickers)

        if self.history_close is None:
            self.history_close = current_close_df
            self.history_volume = current_volume_df
            return pd.Series(0.0, index=tickers)
        else:
            self.history_close = pd.concat([self.history_close, current_close_df])
            self.history_volume = pd.concat([self.history_volume, current_volume_df])

        # 3. 历史数据不足时返回空仓
        if len(self.history_close) < self.min_history:
            return pd.Series(0.0, index=tickers)

        # 4. 计算对数收益率（用于部分因子）
        self.history_ret = np.log(self.history_close / self.history_close.shift(1))
        self.history_ret = self.history_ret.dropna(how='all')

        close_hist = self.history_close
        ret_hist = self.history_ret
        vol_hist = self.history_volume
        latest_idx = close_hist.index[-1]

        # 5. 计算所有原始因子（取最新时刻值）
        raw_signals = {}
        for name in self.factor_names:
            if name == "zprice_12":
                factor_df = self._zprice_n(close_hist, 12)
            elif name == "zprice_36":
                factor_df = self._zprice_n(close_hist, 36)
            elif name == "vol_weighted_momentum_6":
                factor_df = self._vol_weighted_momentum(ret_hist, 6)
            elif name == "volume_adjusted_return_12":
                # 若无有效成交量，跳过
                if vol_hist.isna().all().all():
                    continue
                factor_df = self._volume_adjusted_return(close_hist, vol_hist, 12)
            else:
                continue

            if latest_idx not in factor_df.index:
                continue
            raw_series = factor_df.loc[latest_idx]
            if raw_series.isna().all():
                continue
            raw_signals[name] = raw_series

        if not raw_signals:
            return pd.Series(0.0, index=tickers)

        # 6. 标准化因子（使用训练期全局参数）
        std_signals = {}
        for name, raw_series in raw_signals.items():
            std_signals[name] = self._standardize_factor(
                raw_series, self.factor_mean[name], self.factor_std[name]
            )

        # 7. 获取当前时刻的 IC 权重（向前填充）
        # 将历史时间戳映射到最近可用 IC 权重（由于我们使用整数索引，需基于实际时间对齐）
        # 简便方法：取 ic_weight 的最后一行（假设 IC 权重每日更新一次，测试期内不变）
        # 更严谨的做法是根据当前真实时间对齐，但框架未传递时间戳，故用最新 IC 权重
        ic_t = self.ic_weight.iloc[-1]  # 用训练期最后一天的 IC 权重
        # 若需动态使用，可改为：ic_t = self.ic_weight.reindex([current_time], method='ffill').iloc[0]

        # 8. 合成信号：加权求和，再用绝对 IC 权重之和归一化
        signal = None
        abs_sum = 0.0
        for name in self.factor_names:
            if name not in std_signals:
                continue
            w = ic_t.get(name, 0.0)
            if w == 0:
                continue
            s = std_signals[name]
            if signal is None:
                signal = w * s
            else:
                signal += w * s
            abs_sum += abs(w)

        if signal is None or abs_sum == 0:
            return pd.Series(0.0, index=tickers)

        signal = signal / abs_sum

        # 9. 选股：信号最高的 20% 等权做多
        valid = signal.dropna()
        if len(valid) < 10:
            return pd.Series(0.0, index=tickers)

        n_long = int(np.ceil(len(valid) * 0.2))
        top_stocks = valid.nlargest(n_long).index

        target_weights = pd.Series(0.0, index=tickers)
        target_weights[top_stocks] = 1.0 / n_long

        # 10. 简单的换手率平滑（可选）
        if self.prev_weights is not None:
            turnover = (target_weights - self.prev_weights).abs().sum()
            if turnover > 0.5:  # 限制单次换手不超过50%
                alpha = 0.5 / turnover
                target_weights = self.prev_weights + alpha * (target_weights - self.prev_weights)
                target_weights = target_weights.clip(lower=0)
                s = target_weights.sum()
                if s > 0:
                    target_weights = target_weights / s

        self.prev_weights = target_weights.copy()
        return target_weights
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque

class Strategy:
    def __init__(self):
        # ---------- 配置 ----------
        self.bars_per_day = 78               # 美股每天交易 78 个 5 分钟
        self.lookback_window = 100           # 保留历史数据长度
        self.top_pct = 0.2
        self.selected_factors = [
            'zprice_12',
            'zprice_36',
            'vol_weighted_momentum_6',
            'volume_adjusted_return_12'
        ]
        
        # ---------- 加载训练期 IC 权重，并转换为按交易日对齐 ----------
        ic_path = Path(__file__).parent / "ic_weight.parquet"
        if not ic_path.exists():
            raise FileNotFoundError(f"未找到 IC 权重文件: {ic_path}")
        ic_all = pd.read_parquet(ic_path).astype("float32")
        ic_all = ic_all[self.selected_factors]
        
        # 计算每个交易日最后一个时刻的 IC 权重（代表当日权重）
        self.daily_ic_weights = ic_all.groupby(ic_all.index.normalize()).last()
        self.daily_ic_weights = self.daily_ic_weights.reset_index(drop=True)  # 索引转为整数交易日序号
        
        # ---------- 数据缓冲区 ----------
        self.close_history = {}     # ticker -> deque of prices
        self.volume_history = {}    # ticker -> deque of volumes
        self.ret_history = {}       # ticker -> deque of log returns
        
        # ---------- 交易日状态 ----------
        self.bar_count = 0                      # 总 K 线计数
        self.trading_day_index = -1             # 当前是第几个交易日（从0开始）
        self.is_first_bar_of_day = False        # 当前是否为当日第一分钟
        
        # 日内因子缓存（用于计算截面均值和标准差）
        self.day_factor_cache = {name: [] for name in self.selected_factors}
        self.day_tickers = None                 # 缓存股票列表，确保列对齐
        
    # ==================== 因子计算函数（同前，略作优化） ====================
    def _zprice_n(self, close_df: pd.DataFrame, period: int) -> pd.Series:
        roll = close_df.rolling(period, min_periods=period)
        ma = roll.mean()
        std = roll.std()
        z = (close_df.iloc[-1] - ma.iloc[-1]) / std.iloc[-1]
        z[std.iloc[-1] == 0] = 0.0
        return z
    
    def _vol_weighted_momentum(self, close_df: pd.DataFrame, volume_df: pd.DataFrame, period: int) -> pd.Series:
        vol_prev = volume_df.shift(period).iloc[-1]
        vol_curr = volume_df.iloc[-1]
        close_prev = close_df.shift(period).iloc[-1]
        close_curr = close_df.iloc[-1]
        mask = (vol_prev > 0) & vol_curr.notna() & vol_prev.notna() & close_prev.notna() & (close_prev > 0)
        vol_weight = (vol_curr / vol_prev).where(mask, np.nan)
        ret = ((close_curr - close_prev) / close_prev).where(mask, np.nan)
        return ret * vol_weight
    
    def _volume_adjusted_return(self, close_df: pd.DataFrame, volume_df: pd.DataFrame, vol_lookback: int, ret_lookback: int) -> pd.Series:
        ret = close_df.pct_change()
        vol_ratio = volume_df / volume_df.rolling(vol_lookback, min_periods=vol_lookback//2).mean()
        adj_ret = ret * vol_ratio
        return adj_ret.rolling(ret_lookback, min_periods=ret_lookback//2).sum().iloc[-1]
    
    # ==================== 历史数据维护 ====================
    def _update_history(self, market_data: pd.DataFrame):
        close = market_data['close']
        volume = market_data.get('volume', pd.Series(index=close.index, dtype=float))
        
        for ticker in close.index:
            if ticker not in self.close_history:
                self.close_history[ticker] = deque(maxlen=self.lookback_window)
                self.volume_history[ticker] = deque(maxlen=self.lookback_window)
                self.ret_history[ticker] = deque(maxlen=self.lookback_window)
            
            price = close[ticker]
            vol = volume[ticker] if pd.notna(volume[ticker]) else 0.0
            
            if len(self.close_history[ticker]) > 0:
                prev_price = self.close_history[ticker][-1]
                if prev_price > 0 and pd.notna(price) and price > 0:
                    log_ret = np.log(price / prev_price)
                else:
                    log_ret = np.nan
            else:
                log_ret = np.nan
            
            # 若为当日第一分钟，强制将收益设为 NaN（剔除隔夜跳空）
            if self.is_first_bar_of_day:
                log_ret = np.nan
            
            self.close_history[ticker].append(price)
            self.volume_history[ticker].append(vol)
            self.ret_history[ticker].append(log_ret)
    
    def _build_dataframes(self) -> tuple:
        if not self.close_history:
            return None, None, None
        lengths = [len(v) for v in self.close_history.values()]
        min_len = min(lengths) if lengths else 0
        if min_len == 0:
            return None, None, None
        
        close_dict = {t: list(self.close_history[t])[-min_len:] for t in self.close_history}
        volume_dict = {t: list(self.volume_history[t])[-min_len:] for t in self.volume_history}
        ret_dict = {t: list(self.ret_history[t])[-min_len:] for t in self.ret_history}
        
        idx = range(min_len)
        close_df = pd.DataFrame(close_dict, index=idx)
        volume_df = pd.DataFrame(volume_dict, index=idx)
        ret_df = pd.DataFrame(ret_dict, index=idx)
        return close_df, volume_df, ret_df
    
    # ==================== 日内截面标准化（Z-Score，使用当日累积数据） ====================
    def _normalize_within_day(self, raw_factors: dict) -> dict:
        """用当日已累积的所有截面的均值和标准差对当前时刻进行 Z-Score 标准化"""
        norm_factors = {}
        for name in self.selected_factors:
            cache = self.day_factor_cache[name]  # 列表，每个元素是一个 Series（之前时刻的因子值）
            if len(cache) < 2:   # 至少需要两个截面才能计算标准差
                return None
            # 将缓存转为 DataFrame，行=时刻，列=股票
            df = pd.DataFrame(cache, index=range(len(cache)))
            # 对齐列
            if self.day_tickers is not None:
                df = df.reindex(columns=self.day_tickers, fill_value=np.nan)
            mean = df.mean()
            std = df.std().replace(0, np.nan)
            current = raw_factors[name].reindex(self.day_tickers, fill_value=np.nan)
            z = (current - mean) / std
            z[std.isna()] = 0.0
            norm_factors[name] = z.fillna(0.0)
        return norm_factors
    
    # ==================== 主 step 函数 ====================
    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        self.bar_count += 1
        
        # 判断新交易日开始
        if self.bar_count % self.bars_per_day == 1:
            self.is_first_bar_of_day = True
            self.trading_day_index += 1
            # 清空当日因子缓存
            for name in self.selected_factors:
                self.day_factor_cache[name] = []
            self.day_tickers = current_market_data.index.tolist()
        else:
            self.is_first_bar_of_day = False
        
        # 更新历史数据
        self._update_history(current_market_data)
        
        # 构建 DataFrame
        close_df, volume_df, ret_df = self._build_dataframes()
        if close_df is None or len(close_df) < 36:
            return pd.Series(0.0, index=current_market_data.index)
        
        # 计算原始因子
        factor_raw = {}
        try:
            factor_raw['zprice_12'] = self._zprice_n(close_df, period=12)
            factor_raw['zprice_36'] = self._zprice_n(close_df, period=36)
            factor_raw['vol_weighted_momentum_6'] = self._vol_weighted_momentum(close_df, volume_df, period=6)
            factor_raw['volume_adjusted_return_12'] = self._volume_adjusted_return(close_df, volume_df, vol_lookback=12, ret_lookback=6)
        except Exception:
            return pd.Series(0.0, index=current_market_data.index)
        
        # 将原始因子加入当日缓存
        for name in self.selected_factors:
            self.day_factor_cache[name].append(factor_raw[name])
        
        # 日内截面标准化
        norm_factors = self._normalize_within_day(factor_raw)
        if norm_factors is None:
            return pd.Series(0.0, index=current_market_data.index)
        
        # 获取当前交易日的 IC 权重
        if self.trading_day_index < len(self.daily_ic_weights):
            ic_t = self.daily_ic_weights.iloc[self.trading_day_index]
        else:
            ic_t = self.daily_ic_weights.iloc[-1]  # 若测试期交易日超过训练期，用最后一天权重
        
        # 合成信号
        signal = pd.Series(0.0, index=self.day_tickers)
        abs_sum = 0.0
        for name in self.selected_factors:
            w = ic_t[name]
            if pd.isna(w):
                continue
            f_vals = norm_factors[name].reindex(self.day_tickers, fill_value=0.0)
            signal += w * f_vals
            abs_sum += abs(w)
        
        if abs_sum > 0:
            signal = signal / abs_sum
        else:
            return pd.Series(0.0, index=self.day_tickers)
        
        # 选前 top_pct 股票等权做多
        valid = signal.dropna()
        if len(valid) < 10:
            return pd.Series(0.0, index=self.day_tickers)
        n_long = max(1, int(np.ceil(len(valid) * self.top_pct)))
        long_stocks = valid.sort_values(ascending=False).index[:n_long]
        weights = pd.Series(0.0, index=self.day_tickers)
        weights[long_stocks] = 1.0 / n_long
        return weights
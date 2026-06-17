import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

# ==========================================
# 一、 对齐 TradingView 的指标算法 (RMA & Session VWAP)
# ==========================================

def calc_rma(series, length):
    """对齐 Pine Script 的 ta.rma (Wilder's Smoothing)"""
    alpha = 1.0 / length
    # 第一个值必须是 SMA
    series_s = pd.Series(series)
    sma = series_s.rolling(window=length, min_periods=length).mean()
    rma = np.full_like(series, np.nan, dtype=float)
    
    start_idx = length - 1
    if len(series) > start_idx:
        first_sma = sma.iloc[start_idx]
        if not np.isnan(first_sma):
            rma[start_idx] = first_sma
            for i in range(start_idx + 1, len(series)):
                rma[i] = alpha * series[i] + (1 - alpha) * rma[i-1]
    return rma

def calc_atr_pine(high, low, close, length):
    """完全对齐 Pine Script 的 ta.atr"""
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    tr = pd.concat([
        h - l, 
        (h - c.shift(1)).abs(), 
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    return calc_rma(tr.values, length)

def calc_adx_pine(high, low, close, length):
    """完全对齐 Pine Script 的 ta.dmi/ta.adx"""
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    up = h - h.shift(1)
    down = l.shift(1) - l
    
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    tr_rma = calc_rma(tr.values, length)
    
    plus_di = 100 * calc_rma(plus_dm, length) / tr_rma
    minus_di = 100 * calc_rma(minus_dm, length) / tr_rma
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    return calc_rma(dx, length)

def calc_session_vwap(df):
    """对齐 TradingView 的 ta.vwap (每日 00:00 重置)"""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tpv = tp * df['Volume']
    vol = df['Volume']
    
    # 按照日期分组并计算累计值
    vwap = tpv.groupby(df.index.date, group_keys=False).cumsum() / \
           vol.groupby(df.index.date, group_keys=False).cumsum()
    return vwap.values

# ==========================================
# 二、 策略类
# ==========================================

class AMEliteStrategy(Strategy):
    ema_filter_len = 800  # 4H 200 -> 1H 800
    kc_len = 20
    kc_mult = 2.6
    atr_len = 14
    vol_mult = 1.5
    adx_th = 15
    risk_pct = 2.0
    be_trigger = 1.8
    tp_part_trig = 3.0
    trail_dist = 2.5

    def init(self):
        high, low, close, vol = self.data.High, self.data.Low, self.data.Close, self.data.Volume

        # 核心指标
        self.ema4h = self.I(lambda x: pd.Series(x).ewm(span=self.ema_filter_len, adjust=False).mean(), close)
        self.ma20 = self.I(lambda x: pd.Series(x).ewm(span=self.kc_len, adjust=False).mean(), close)
        self.atr14 = self.I(calc_atr_pine, high, low, close, self.atr_len)
        self.adx = self.I(calc_adx_pine, high, low, close, 14)
        self.vol_ma = self.I(lambda x: pd.Series(x).rolling(20).mean(), vol)
        self.vwap = self.I(lambda: self.data.df['VWAP_precalc'].values)

        # 状态机
        self.current_sl = None
        self.max_high = None
        self.min_low = None
        self.entry_atr = None
        self.entry_price = None

    def next(self):
        if len(self.data) < self.ema_filter_len: return

        price, high, low, vol = self.data.Close[-1], self.data.High[-1], self.data.Low[-1], self.data.Volume[-1]
        dt = self.data.index[-1]
        
        atr = self.atr14[-1]
        kc_u = self.ma20[-1] + (self.kc_mult * atr)
        kc_l = self.ma20[-1] - (self.kc_mult * atr)

        if self.position:
            # 动态风控逻辑
            if self.position.is_long:
                self.max_high = max(self.max_high or high, high)
                # 保本触发
                if high >= self.entry_price + (self.entry_atr * self.be_trigger):
                    self.current_sl = max(self.current_sl or 0, self.entry_price)
                # 追踪触发
                if self.max_high >= self.entry_price + (self.entry_atr * self.tp_part_trig):
                    trail_val = self.max_high - (atr * self.trail_dist)
                    self.current_sl = max(self.current_sl or 0, trail_val)
            else:
                self.min_low = min(self.min_low or low, low)
                if low <= self.entry_price - (self.entry_atr * self.be_trigger):
                    self.current_sl = min(self.current_sl or 999999, self.entry_price)
                if self.min_low <= self.entry_price - (self.entry_atr * self.tp_part_trig):
                    trail_val = self.min_low + (atr * self.trail_dist)
                    self.current_sl = min(self.current_sl or 999999, trail_val)

            for trade in self.trades:
                trade.sl = self.current_sl
        else:
            # 入场条件
            is_weekend = (dt.dayofweek == 4 and dt.hour >= 16) or (dt.dayofweek in [5, 6])
            vol_cond = vol > (self.vol_ma[-1] * self.vol_mult)
            trend_strength = self.adx[-1] > self.adx_th

            long_cond = price > self.ema4h[-1] and price > kc_u and vol_cond and price > self.vwap[-1] and trend_strength and not is_weekend
            short_cond = price < self.ema4h[-1] and price < kc_l and vol_cond and price < self.vwap[-1] and trend_strength and not is_weekend

            if long_cond: self._open(price, high, True)
            elif short_cond: self._open(price, low, False)

    def _open(self, price, extreme, is_long):
        self.entry_atr, self.entry_price = self.atr14[-1], price
        stop_dist = 2 * self.entry_atr
        risk_amt = self.equity * (self.risk_pct / 100)
        size = min(0.95, (risk_amt / stop_dist * price) / self.equity)

        if is_long:
            self.max_high, self.current_sl = extreme, price - stop_dist
            tp = price + (self.entry_atr * self.tp_part_trig)
            self.buy(size=size/2, sl=self.current_sl, tp=tp) 
            self.buy(size=size/2, sl=self.current_sl)        
        else:
            self.min_low, self.current_sl = extreme, price + stop_dist
            tp = price - (self.entry_atr * self.tp_part_trig)
            self.sell(size=size/2, sl=self.current_sl, tp=tp)
            self.sell(size=size/2, sl=self.current_sl)

if __name__ == '__main__':
    df = pd.read_csv('ETHUSDT_2025_2026_1h.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df['VWAP_precalc'] = calc_session_vwap(df)

    bt = Backtest(df, AMEliteStrategy, cash=10000, commission=0.0004)
    stats = bt.run()
    print(stats)

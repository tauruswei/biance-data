import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from backtesting import Backtest, Strategy

# ==========================================
# Indicators Support Functions
# ==========================================

def calc_rma(series, length):
    alpha = 1.0 / length
    series_s = pd.Series(series)
    sma = series_s.rolling(window=length, min_periods=length).mean()
    rma = np.full_like(series, np.nan, dtype=float)
    start_idx = length - 1
    if len(series) > start_idx:
        rma[start_idx] = sma.iloc[start_idx]
        for i in range(start_idx + 1, len(series)):
            rma[i] = alpha * series[i] + (1 - alpha) * rma[i-1]
    return rma

def calc_atr_pine(high, low, close, length):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return calc_rma(tr.values, length)

def calc_adx(high, low, close, length):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    up = h - h.shift(1)
    down = l.shift(1) - l
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = calc_rma(tr.values, length)
    plus_di = 100 * calc_rma(plus_dm, length) / atr
    minus_di = 100 * calc_rma(minus_dm, length) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.nan_to_num(dx, nan=0.0)
    adx = calc_rma(dx, length)
    return adx

# ==========================================
# Strategy: Sentinel-5.3 (Dynamic Adaptive Trailing)
# ==========================================

class Sentinel5Adaptive(Strategy):
    ema_filter_len = 400
    kc_mult = 1.8
    adx_threshold = 20
    risk_pct = 1.5
    
    # Asymmetric Parameters
    tp_mult_long = 9.0
    tp_mult_short = 3.0
    sl_mult_long = 2.4
    sl_mult_short = 2.4
    exit_span_long = 200
    exit_span_short = 15
    
    trailing_sl_mult_long = 2.5
    
    # Dynamic mode
    # 'A': Scheme A (Conservative), 'B': Scheme B (Explosive), 'C': Scheme C (Dynamic Adaptive)
    mode = 'C'
    
    # Static parameters if not in Scheme C
    vol_mult = 2.2
    p_tp1_long = 0.2
    
    def init(self):
        high, low, close, vol = self.data.High, self.data.Low, self.data.Close, self.data.Volume
        self.ema_trend = self.I(lambda x: pd.Series(x).ewm(span=self.ema_filter_len, adjust=False).mean(), close)
        self.ma_kc_long = self.I(lambda x: pd.Series(x).ewm(span=self.exit_span_long, adjust=False).mean(), close)
        self.ma_kc_short = self.I(lambda x: pd.Series(x).ewm(span=self.exit_span_short, adjust=False).mean(), close)
        self.atr = self.I(calc_atr_pine, high, low, close, 14)
        self.vol_ma = self.I(lambda x: pd.Series(x).rolling(20).mean(), vol)
        self.adx = self.I(calc_adx, high, low, close, 14)
        
        macd_line = pd.Series(close).ewm(span=12, adjust=False).mean() - pd.Series(close).ewm(span=26, adjust=False).mean()
        self.macd = self.I(lambda x: macd_line, close)
        self.macd_signal = self.I(lambda x: macd_line.ewm(span=9, adjust=False).mean(), close)

        self.tp1_hit = False
        self.entry_price = 0
        self.entry_atr = 0
        self.highest_price = 0
        self.dynamic_p_tp1 = 0.2

    def next(self):
        if len(self.data) < self.ema_filter_len: return

        price = self.data.Close[-1]
        atr = self.atr[-1]
        adx = self.adx[-1]
        
        if self.position:
            if self.position.is_long:
                if not self.tp1_hit:
                    if self.data.High[-1] >= self.entry_price + (self.entry_atr * self.tp_mult_long):
                        self.tp1_hit = True
                        self.highest_price = self.data.High[-1]
                        new_sl = self.entry_price + (self.entry_atr * 0.5)
                        for trade in self.trades:
                            trade.sl = new_sl
                else:
                    self.highest_price = max(self.highest_price, self.data.High[-1])
                    new_sl = max(self.entry_price + (self.entry_atr * 0.5), self.highest_price - self.trailing_sl_mult_long * atr)
                    for trade in self.trades:
                        if trade.sl is None or new_sl > trade.sl:
                            trade.sl = new_sl
                            
                if self.tp1_hit and price < self.ma_kc_long[-1]:
                    self.position.close()
            
            elif self.position.is_short:
                if not self.tp1_hit and self.data.Low[-1] <= self.entry_price - (self.entry_atr * self.tp_mult_short):
                    self.tp1_hit = True
                    new_sl = self.entry_price - (self.entry_atr * 0.5)
                    for trade in self.trades:
                        trade.sl = new_sl
                if self.tp1_hit and price > self.ma_kc_short[-1]:
                    self.position.close()
        else:
            # Adaptive threshold logic
            if self.mode == 'C':
                if adx > 25:  # Strong trend state
                    current_vol_mult = 1.8
                    self.dynamic_p_tp1 = 0.3
                else:         # Quiet or sideways state
                    current_vol_mult = self.vol_mult
                    self.dynamic_p_tp1 = self.p_tp1_long
            else:
                current_vol_mult = self.vol_mult
                self.dynamic_p_tp1 = self.p_tp1_long

            vol_cond = self.data.Volume[-1] > (self.vol_ma[-1] * current_vol_mult)
            adx_cond = adx > self.adx_threshold
            macd_up = self.macd[-1] > self.macd_signal[-1]
            macd_down = self.macd[-1] < self.macd_signal[-1]
            
            effective_kc_mult = self.kc_mult if adx < 30 else self.kc_mult * 0.8
            
            long_cond = (price > self.ema_trend[-1] and 
                          price > self.ma_kc_long[-1] + (effective_kc_mult * atr) and 
                          vol_cond and adx_cond and macd_up)
            
            short_cond = (price < self.ema_trend[-1] and 
                          price < self.ma_kc_short[-1] - (effective_kc_mult * atr) and 
                          vol_cond and adx_cond and macd_down)
            
            if long_cond: self._open(price, True)
            elif short_cond: self._open(price, False)

    def _open(self, price, is_long):
        self.entry_price = price
        self.entry_atr = self.atr[-1]
        if self.entry_atr == 0: return
        self.tp1_hit = False
        
        if is_long:
            self.highest_price = price
            stop_dist = self.sl_mult_long * self.entry_atr
            risk_amt = self.equity * (self.risk_pct / 100)
            total_size_pct = min(0.9, (risk_amt / stop_dist * price) / self.equity)
            if total_size_pct < 0.05: return
            
            size1 = total_size_pct * self.dynamic_p_tp1
            size2 = (total_size_pct - size1) / (1 - size1)
            
            sl, tp = price - stop_dist, price + (self.entry_atr * self.tp_mult_long)
            if size1 > 0:
                self.buy(size=size1, sl=sl, tp=tp)
            if size2 > 0:
                self.buy(size=size2, sl=sl)
        else:
            stop_dist = self.sl_mult_short * self.entry_atr
            risk_amt = self.equity * (self.risk_pct / 100)
            total_size_pct = min(0.9, (risk_amt / stop_dist * price) / self.equity)
            if total_size_pct < 0.05: return
            size1 = total_size_pct / 2
            size2 = size1 / (1 - size1)
            sl, tp = price + stop_dist, price - (self.entry_atr * self.tp_mult_short)
            self.sell(size=size1, sl=sl, tp=tp)
            self.sell(size=size2, sl=sl)

# ==========================================
# Main Execution: Full Period Backtest (2017-2025)
# ==========================================

if __name__ == '__main__':
    files = [
        'BTCUSDT_2017_1h.csv', 'BTCUSDT_2018_1h.csv', 'BTCUSDT_2019_1h.csv',
        'BTCUSDT_2020_1h.csv', 'BTCUSDT_2021_1h.csv', 'BTCUSDT_2022_1h.csv',
        'BTCUSDT_2023_1h.csv', 'BTCUSDT_2024_1h.csv', 'BTCUSDT_2025_1h.csv'
    ]
    
    all_df = []
    for f in files:
        if os.path.exists(f):
            all_df.append(pd.read_csv(f))
    
    df = pd.concat(all_df).drop_duplicates(subset=['Timestamp']).sort_values('Timestamp')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    df.set_index('Timestamp', inplace=True)
    
    print(f"Combined data from {df.index[0]} to {df.index[-1]}")

    modes = {
        'Scheme_A': {'mode': 'A', 'vol_mult': 2.2, 'p_tp1_long': 0.2, 'desc': 'Ultra-low frequency'},
        'Scheme_B': {'mode': 'B', 'vol_mult': 1.8, 'p_tp1_long': 0.3, 'desc': 'Maximum explosive'},
        'Scheme_C': {'mode': 'C', 'vol_mult': 2.2, 'p_tp1_long': 0.2, 'desc': 'Adaptive Optimal Hybrid'}
    }

    results = {}

    for name, config in modes.items():
        print(f"\n--- Running Full Period Backtest for {name} ({config['desc']}) ---")
        Sentinel5Adaptive.mode = config['mode']
        Sentinel5Adaptive.vol_mult = config['vol_mult']
        Sentinel5Adaptive.p_tp1_long = config['p_tp1_long']
        
        bt = Backtest(df, Sentinel5Adaptive, cash=10000000, commission=0.0004)
        stats = bt.run()
        results[name] = stats
        
        # Save Scheme_C trades to btc_full_period_trades.csv by default
        if name == 'Scheme_C':
            stats['_trades'].to_csv('btc_full_period_trades.csv')

    print("\n" + "="*80)
    print("=== BTC Long-term Compounding Results Comparison (2017-2025) ===")
    print("="*80)
    
    metrics = [
        ('Start Date', lambda s: s['Start']),
        ('End Date', lambda s: s['End']),
        ('Initial Capital', lambda s: '$10,000,000'),
        ('Final Equity', lambda s: f"${s['Equity Final [$]']:.2f}"),
        ('Total Return [%]', lambda s: f"{s['Return [%]']:.2f}%"),
        ('CAGR [%]', lambda s: f"{s['CAGR [%]']:.2f}%"),
        ('Max Drawdown [%]', lambda s: f"{s['Max. Drawdown [%]']:.2f}%"),
        ('Sharpe Ratio', lambda s: f"{s['Sharpe Ratio']:.2f}"),
        ('Sortino Ratio', lambda s: f"{s['Sortino Ratio']:.2f}"),
        ('Total Trades', lambda s: s['# Trades']),
        ('Win Rate [%]', lambda s: f"{s['Win Rate [%]']:.2f}%"),
        ('Profit Factor', lambda s: f"{s['Profit Factor']:.2f}"),
    ]

    comp_data = []
    for m_name, getter in metrics:
        comp_data.append({
            'Metric': m_name,
            'Scheme A (Ultra-low Freq)': getter(results['Scheme_A']),
            'Scheme B (Max Explosive)': getter(results['Scheme_B']),
            'Scheme C (Adaptive Optimal)': getter(results['Scheme_C'])
        })
    
    comp_df = pd.DataFrame(comp_data)
    print(comp_df.to_string(index=False))

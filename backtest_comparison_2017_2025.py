import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from backtesting import Backtest, Strategy

# Indicators functions
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

# Base Strategy
class Sentinel5Base(Strategy):
    ema_filter_len = 400
    kc_mult = 1.6
    adx_threshold = 18
    risk_pct = 1.5
    
    tp_mult_long = 9.0
    tp_mult_short = 3.0
    sl_mult_long = 2.2
    sl_mult_short = 2.2
    exit_span_long = 200
    exit_span_short = 15
    trailing_sl_mult_long = 2.5
    
    mode = 'C'
    vol_mult = 2.4
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
            if self.mode == 'C':
                if adx > 25:
                    current_vol_mult = 1.8
                    self.dynamic_p_tp1 = 0.3
                else:
                    current_vol_mult = self.get_quiet_vol_mult()
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

    def get_quiet_vol_mult(self):
        return self.vol_mult

# ETH strategy
class EthStrategy(Sentinel5Base):
    kc_mult = 1.6
    adx_threshold = 18
    sl_mult_long = 2.2
    sl_mult_short = 2.2
    vol_mult = 2.4
    
    def get_quiet_vol_mult(self):
        return 2.4 # ETH uses hardcoded 2.4 in Scheme C quiet state

# BTC strategy
class BtcStrategy(Sentinel5Base):
    kc_mult = 1.8
    adx_threshold = 20
    sl_mult_long = 2.4
    sl_mult_short = 2.4
    vol_mult = 2.2
    
    def get_quiet_vol_mult(self):
        return self.vol_mult # BTC uses dynamic vol_mult

# Main comparison
if __name__ == '__main__':
    years = range(2017, 2026)
    btc_data = {}
    eth_data = {}
    
    for year in years:
        # Load BTC
        f_btc = f'BTCUSDT_{year}_1h.csv'
        if os.path.exists(f_btc):
            df = pd.read_csv(f_btc)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
            df.set_index('Timestamp', inplace=True)
            btc_data[year] = df
        
        # Load ETH
        f_eth = f'ETHUSDT_{year}_1h.csv'
        if os.path.exists(f_eth):
            df = pd.read_csv(f_eth)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
            df.set_index('Timestamp', inplace=True)
            eth_data[year] = df
            
    # Run yearly
    yearly_results = []
    for year in years:
        btc_ret, eth_ret = 0.0, 0.0
        btc_trades, eth_trades = 0, 0
        btc_dd, eth_dd = 0.0, 0.0
        btc_bh, eth_bh = 0.0, 0.0
        
        # Run BTC
        if year in btc_data:
            bt = Backtest(btc_data[year], BtcStrategy, cash=10000000, commission=0.0004)
            s = bt.run()
            btc_ret = s['Return [%]']
            btc_trades = s['# Trades']
            btc_dd = s['Max. Drawdown [%]']
            btc_bh = s['Buy & Hold Return [%]']
            
        # Run ETH
        if year in eth_data:
            bt = Backtest(eth_data[year], EthStrategy, cash=10000000, commission=0.0004)
            s = bt.run()
            eth_ret = s['Return [%]']
            eth_trades = s['# Trades']
            eth_dd = s['Max. Drawdown [%]']
            eth_bh = s['Buy & Hold Return [%]']
            
        yearly_results.append({
            'Year': year,
            'BTC_B&H [%]': round(btc_bh, 2),
            'BTC_Return [%]': round(btc_ret, 2),
            'BTC_Trades': btc_trades,
            'BTC_MaxDD [%]': round(btc_dd, 2),
            'ETH_B&H [%]': round(eth_bh, 2),
            'ETH_Return [%]': round(eth_ret, 2),
            'ETH_Trades': eth_trades,
            'ETH_MaxDD [%]': round(eth_dd, 2)
        })
        
    yearly_df = pd.DataFrame(yearly_results)
    print("\n=== BTC vs ETH Yearly Performance Comparison (Scheme C, 2017-2025) ===")
    print(yearly_df.to_string(index=False))
    yearly_df.to_csv('yearly_comparison_btc_eth_2017_2025.csv', index=False)
    
    # Run continuous compounding
    btc_all = []
    eth_all = []
    for year in years:
        if year in btc_data: btc_all.append(btc_data[year])
        if year in eth_data: eth_all.append(eth_data[year])
        
    btc_df = pd.concat(btc_all).drop_duplicates().sort_index()
    eth_df = pd.concat(eth_all).drop_duplicates().sort_index()
    
    bt_btc = Backtest(btc_df, BtcStrategy, cash=10000000, commission=0.0004)
    s_btc = bt_btc.run()
    
    bt_eth = Backtest(eth_df, EthStrategy, cash=10000000, commission=0.0004)
    s_eth = bt_eth.run()
    
    print("\n=== BTC vs ETH Long-term Compounding Comparison (Scheme C, 2017-2025) ===")
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
            'BTC (Scheme C)': getter(s_btc),
            'ETH (Scheme C)': getter(s_eth)
        })
    comp_df = pd.DataFrame(comp_data)
    print(comp_df.to_string(index=False))

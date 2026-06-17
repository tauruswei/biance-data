import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

# ==========================================
# 一、 辅助计算函数 (RMA & Pine Script 指标)
# ==========================================

def calc_rma(series, length):
    """对齐 Pine Script 的 ta.rma (Wilder's Smoothing)"""
    alpha = 1.0 / length
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

# ==========================================
# 二、 Sentinel-5 哨兵系统综合策略
# ==========================================

class Sentinel5Strategy(Strategy):
    # 策略参数
    ema_filter_len = 800  # 4H EMA 200 在 1H 周期上的等效替代 (4 * 200)
    kc_len = 20           # 肯特纳通道 EMA 周期
    kc_mult = 2.25        # 肯特纳通道 ATR 乘数
    atr_len = 14          # ATR 周期
    vol_mult = 1.5        # 突破量能要求 (大于 20日均量 1.5 倍)
    
    # 风控与资金管理参数
    risk_pct = 1.5        # 单笔交易允许亏损的总资金百分比
    tp_mult = 1.5         # 一级止盈倍数 (1.5 倍 ATR 止盈一半)
    sl_mult = 2.0         # 初始止损倍数 (2 倍 ATR 宽止损防插针)

    def init(self):
        high, low, close, vol = self.data.High, self.data.Low, self.data.Close, self.data.Volume

        # 大趋势过滤: 4H 级别 EMA 200 (1H 上的 800 EMA)
        self.ema4h = self.I(lambda x: pd.Series(x).ewm(span=self.ema_filter_len, adjust=False).mean(), close)
        
        # 动量入场中轨: 1H 级别 EMA 20
        self.ma20 = self.I(lambda x: pd.Series(x).ewm(span=self.kc_len, adjust=False).mean(), close)
        
        # 波动率衡量: ATR
        self.atr14 = self.I(calc_atr_pine, high, low, close, self.atr_len)
        
        # 量能辅助: Volume MA 20
        self.vol_ma = self.I(lambda x: pd.Series(x).rolling(20).mean(), vol)

        # 订单状态管理
        self.entry_price = None
        self.entry_atr = None
        self.tp1_hit = False
        self.current_sl = None

    def next(self):
        # 预热期过滤
        if len(self.data) < self.ema_filter_len: 
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]
        
        atr = self.atr14[-1]
        
        # KC 肯特纳通道上下轨计算
        kc_u = self.ma20[-1] + (self.kc_mult * atr)
        kc_l = self.ma20[-1] - (self.kc_mult * atr)

        if self.position:
            # === 持仓中的风控与出场逻辑 ===
            if self.position.is_long:
                # 1. 一级止盈与保本 (触及 1.5*ATR 利润)
                if not self.tp1_hit and high >= self.entry_price + (self.entry_atr * self.tp_mult):
                    self.tp1_hit = True
                    self.current_sl = self.entry_price  # 移至成本价
                    for trade in self.trades:
                        trade.sl = self.current_sl
                
                # 2. 二级止盈 (剩余 50% 跌破 EMA 20 离场)
                if self.tp1_hit and price < self.ma20[-1]:
                    self.position.close()
            
            elif self.position.is_short:
                # 1. 一级止盈与保本 (触及 1.5*ATR 利润)
                if not self.tp1_hit and low <= self.entry_price - (self.entry_atr * self.tp_mult):
                    self.tp1_hit = True
                    self.current_sl = self.entry_price  # 移至成本价
                    for trade in self.trades:
                        trade.sl = self.current_sl
                
                # 2. 二级止盈 (剩余 50% 突破 EMA 20 离场)
                if self.tp1_hit and price > self.ma20[-1]:
                    self.position.close()
        else:
            # === 空仓时的入场逻辑 ===
            # 量能真突破确认
            vol_cond = vol > (self.vol_ma[-1] * self.vol_mult)

            # 多头触发: 在大周期均线上方，且收盘价强力突破通道上轨
            long_cond = price > self.ema4h[-1] and price > kc_u and vol_cond
            
            # 空头触发: 在大周期均线下方，且收盘价强力跌破通道下轨
            short_cond = price < self.ema4h[-1] and price < kc_l and vol_cond

            if long_cond:
                self._open(price, True)
            elif short_cond:
                self._open(price, False)

    def _open(self, price, is_long):
        """执行开仓逻辑：动态仓位计算 + 分批挂单"""
        self.entry_atr = self.atr14[-1]
        self.entry_price = price
        self.tp1_hit = False
        
        # 1. 动态仓位计算 (Kelly / 风险百分比模型)
        stop_dist = self.sl_mult * self.entry_atr
        risk_amt = self.equity * (self.risk_pct / 100)
        
        # 目标全仓位大小 (价值占比) = (单笔可承受亏损额 / 每单位亏损) * 当前单价 / 总资金
        # 增加 0.95 的上限，留出 5% 空余避免 Backtesting 的保证金不足报错
        target_size_pct = min(0.95, (risk_amt / stop_dist * price) / self.equity)
        
        if target_size_pct <= 0.01:
            return # 风险过高，ATR太大，放弃这笔交易或仓位极小

        # 将订单一分为二以实现 50% 的一级止盈
        # Backtesting 的 size 参数是基于 *剩余可用可用资金* 的比例，因此需要通过数学换算使两笔单子等大
        size1 = target_size_pct / 2
        size2 = size1 / (1 - size1)

        if is_long:
            self.current_sl = price - stop_dist
            tp1_price = price + (self.entry_atr * self.tp_mult)
            # 第一笔：带固定止盈目标（到达后自动平掉这 50%）
            self.buy(size=size1, sl=self.current_sl, tp=tp1_price)
            # 第二笔：不带固定止盈，靠 EMA20 追踪
            self.buy(size=size2, sl=self.current_sl)
        else:
            self.current_sl = price + stop_dist
            tp1_price = price - (self.entry_atr * self.tp_mult)
            # 第一笔：带固定止盈目标
            self.sell(size=size1, sl=self.current_sl, tp=tp1_price)
            # 第二笔：不带固定止盈，靠 EMA20 追踪
            self.sell(size=size2, sl=self.current_sl)

if __name__ == '__main__':
    df = pd.read_csv('ETHUSDT_2024_1h.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    bt = Backtest(df, Sentinel5Strategy, cash=10000, commission=0.0004)
    stats = bt.run()
    print("="*40)
    print("Sentinel-5 哨兵系统回测结果")
    print("="*40)
    print(stats)
    
    # 打印前 5 笔交易以供检查逻辑
    if not stats._trades.empty:
        print("\n=== 交易明细 (前 10 笔) ===")
        print(stats._trades.head(10).to_string())

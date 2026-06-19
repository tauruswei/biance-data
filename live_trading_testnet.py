import time
import os
import pandas as pd
import numpy as np
import ccxt
import warnings
warnings.filterwarnings("ignore")

# =========================================================================
# Indicators (Must match backtest exactly)
# =========================================================================

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

def compute_indicators(df):
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    vol = df['Volume'].values
    
    df['ema_trend'] = pd.Series(close).ewm(span=400, adjust=False).mean()
    df['ma_kc_long'] = pd.Series(close).ewm(span=200, adjust=False).mean()
    df['ma_kc_short'] = pd.Series(close).ewm(span=15, adjust=False).mean()
    
    df['atr'] = calc_atr_pine(high, low, close, 14)
    df['vol_ma'] = pd.Series(vol).rolling(20).mean()
    df['adx'] = calc_adx(high, low, close, 14)
    
    macd_line = pd.Series(close).ewm(span=12, adjust=False).mean() - pd.Series(close).ewm(span=26, adjust=False).mean()
    df['macd'] = macd_line
    df['macd_signal'] = macd_line.ewm(span=9, adjust=False).mean()
    return df

# =========================================================================
# Live Trading Bot (Binance Futures Testnet)
# =========================================================================

class BinanceTestnetBot:
    def __init__(self, api_key, api_secret, symbol="ETH/USDT", risk_pct=1.5, proxy=None):
        self.symbol = symbol
        self.risk_pct = risk_pct
        
        exchange_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future', # Deploy to Futures (multidirectional)
            }
        }
        
        # Read proxy from parameter, BINANCE_PROXY, HTTP_PROXY, or HTTPS_PROXY environment variables
        proxy_url = proxy or os.getenv("BINANCE_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        if proxy_url:
            exchange_config['proxies'] = {
                'http': proxy_url,
                'https': proxy_url,
            }
            print(f"Using proxy for Binance API connections: {proxy_url}")
            
        # Initialize CCXT Binance connector
        self.exchange = ccxt.binance(exchange_config)
        
        # Enable Demo Trading Mode
        self.exchange.enable_demo_trading(True)
        print("Demo Trading Mode Enabled.")
        
        # Determine strategy parameters based on symbol
        if "BTC" in self.symbol:
            self.adx_threshold = 20
            self.kc_mult = 1.8
            self.sl_mult_long = 2.4
            self.sl_mult_short = 2.4
            self.vol_mult_quiet = 2.2
        else:
            self.adx_threshold = 18
            self.kc_mult = 1.6
            self.sl_mult_long = 2.2
            self.sl_mult_short = 2.2
            self.vol_mult_quiet = 2.4
            
        # State variables (persisted in live trading)
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.highest_price = 0.0
        self.tp1_hit = False
        self.current_sl = 0.0
        self.p_tp1_long = 0.2

    def fetch_market_data(self):
        """Fetch historical K-lines (limit=500, timeframe=1h)"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='1h', limit=500)
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            return compute_indicators(df)
        except Exception as e:
            print(f"Error fetching K-lines: {e}")
            return None

    def get_account_equity(self):
        """Fetch USDB account balance (USDT)"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['total']['USDT']
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0

    def get_open_positions(self):
        """Check if we have an active position"""
        try:
            positions = self.exchange.fetch_positions(symbols=[self.symbol])
            if positions:
                pos = positions[0]
                size = float(pos['contracts'])
                side = pos['side'] # 'long' or 'short'
                entry = float(pos['entryPrice'])
                return {'size': size, 'side': side, 'entryPrice': entry}
            return None
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return None

    def execute_logic(self):
        print(f"\n--- Checking Strategy Rules at {pd.Timestamp.now()} ---")
        df = self.fetch_market_data()
        if df is None or len(df) < 400:
            print("Insufficient data.")
            return
            
        last_row = df.iloc[-1]
        price = last_row['Close']
        atr = last_row['atr']
        adx = last_row['adx']
        vol = last_row['Volume']
        vol_ma = last_row['vol_ma']
        
        ema_trend = last_row['ema_trend']
        ma_kc_long = last_row['ma_kc_long']
        ma_kc_short = last_row['ma_kc_short']
        macd = last_row['macd']
        macd_signal = last_row['macd_signal']
        
        # Check active position
        position = self.get_open_positions()
        equity = self.get_account_equity()
        print(f"Equity: ${equity:.2f} | Last {self.symbol} Price: ${price:.2f}")
        
        if position and position['size'] > 0:
            side = position['side']
            size = position['size']
            print(f"Active Position detected: {side.upper()} | Size: {size} contracts | Entry: ${position['entryPrice']}")
            
            if side == 'long':
                # Update highest price
                self.highest_price = max(self.highest_price, last_row['High'])
                
                # Check TP1 hit
                if not self.tp1_hit:
                    if last_row['High'] >= self.entry_price + (self.entry_atr * 9.0):
                        print("TP1 hit triggered on Long position! Moving SL to break-even + 0.5 * ATR.")
                        self.tp1_hit = True
                        self.current_sl = self.entry_price + (self.entry_atr * 0.5)
                        # In futures testnet, cancel the original SL order and place the new trailing SL order here.
                else:
                    # Trailing Chandelier Stop-loss
                    new_sl = max(self.entry_price + (self.entry_atr * 0.5), self.highest_price - 2.5 * atr)
                    if new_sl > self.current_sl:
                        print(f"Trailing SL raised to: ${new_sl:.2f}")
                        self.current_sl = new_sl
                
                # MA Exit Rule
                if self.tp1_hit and price < ma_kc_long:
                    print("Price fell below exit MA after TP1. Closing remaining position.")
                    self.exchange.create_market_sell_order(self.symbol, size)
                    self.reset_state()
            
            elif side == 'short':
                # Standard Short SL/TP checks
                if not self.tp1_hit:
                    if last_row['Low'] <= self.entry_price - (self.entry_atr * 3.0):
                        print("TP1 hit triggered on Short position! Moving SL to break-even - 0.5 * ATR.")
                        self.tp1_hit = True
                        self.current_sl = self.entry_price - (self.entry_atr * 0.5)
                
                if self.tp1_hit and price > ma_kc_short:
                    print("Price rose above exit MA after TP1. Closing Short position.")
                    self.exchange.create_market_buy_order(self.symbol, size)
                    self.reset_state()
                    
        else:
            # Entry logic
            # Scheme C Dynamic parameters
            if adx > 25:
                vol_mult = 1.8
                self.dynamic_p_tp1 = 0.3
            else:
                vol_mult = self.vol_mult_quiet
                self.dynamic_p_tp1 = 0.2

            vol_cond = vol > (vol_ma * vol_mult)
            adx_cond = adx > self.adx_threshold
            macd_up = macd > macd_signal
            macd_down = macd < macd_signal
            
            effective_kc_mult = self.kc_mult if adx < 30 else self.kc_mult * 0.8
            
            long_cond = (price > ema_trend and 
                          price > ma_kc_long + (effective_kc_mult * atr) and 
                          vol_cond and adx_cond and macd_up)
            
            short_cond = (price < ema_trend and 
                          price < ma_kc_short - (effective_kc_mult * atr) and 
                          vol_cond and adx_cond and macd_down)
            
            if long_cond:
                print(">>> BUY SIGNAL TRIGGERED <<<")
                self.open_position(price, atr, is_long=True, equity=equity)
            elif short_cond:
                print(">>> SELL SIGNAL TRIGGERED <<<")
                self.open_position(price, atr, is_long=False, equity=equity)
            else:
                print("No trading signals triggered.")

    def open_position(self, price, atr, is_long, equity):
        self.entry_price = price
        self.entry_atr = atr
        self.tp1_hit = False
        
        if is_long:
            self.highest_price = price
            stop_dist = self.sl_mult_long * self.entry_atr
            risk_amt = equity * (self.risk_pct / 100)
            
            # Position sizing (based on SL distance)
            total_contracts = risk_amt / stop_dist
            if total_contracts * price > equity * 10: # Safety cap (max 10x leverage)
                total_contracts = (equity * 10) / price
                
            print(f"Calculated target size: {total_contracts:.3f} contracts.")
            
            # Divide into TP1 (limit taker) and Trailing (runs dynamically)
            size1 = total_contracts * self.dynamic_p_tp1
            size2 = total_contracts - size1
            
            # Open Long Orders
            self.exchange.create_market_buy_order(self.symbol, total_contracts)
            print(f"Opened Long Position of {total_contracts:.3f} contracts.")
            
            # Place initial SL order on exchange
            sl_price = price - stop_dist
            self.current_sl = sl_price
            self.exchange.create_order(self.symbol, 'STOP_MARKET', 'sell', total_contracts, None, params={'stopPrice': sl_price, 'reduceOnly': True})
            print(f"Set initial Stop-Loss order at: ${sl_price:.2f}")
            
            # Place TP1 Limit Order
            tp_price = price + (atr * 9.0)
            self.exchange.create_order(self.symbol, 'limit', 'sell', size1, tp_price)
            print(f"Set Take-Profit limit order (for {size1:.3f} contracts) at: ${tp_price:.2f}")
            
        else:
            stop_dist = self.sl_mult_short * self.entry_atr
            risk_amt = equity * (self.risk_pct / 100)
            total_contracts = risk_amt / stop_dist
            if total_contracts * price > equity * 10:
                total_contracts = (equity * 10) / price
                
            self.exchange.create_market_sell_order(self.symbol, total_contracts)
            print(f"Opened Short Position of {total_contracts:.3f} contracts.")
            
            sl_price = price + stop_dist
            self.current_sl = sl_price
            self.exchange.create_order(self.symbol, 'STOP_MARKET', 'buy', total_contracts, None, params={'stopPrice': sl_price, 'reduceOnly': True})
            
            # Place TP1 Limit Order
            tp_price = price - (atr * 3.0)
            self.exchange.create_order(self.symbol, 'limit', 'buy', total_contracts / 2, tp_price)
            print(f"Set Take-Profit limit order at: ${tp_price:.2f}")

    def reset_state(self):
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.highest_price = 0.0
        self.tp1_hit = False
        self.current_sl = 0.0

    def start_polling(self):
        """Daemon runner loop: execute once every hour"""
        print("Bot poller successfully started. Listening to market conditions...")
        while True:
            try:
                self.execute_logic()
            except Exception as e:
                print(f"Error in execution loop: {e}")
            
            # Wait for the next hourly close (approx 60 min sleep)
            # Sleep till next hour start + 10 seconds buffer
            now = time.time()
            sleep_sec = 3600 - (now % 3600) + 10
            print(f"Sleeping for {sleep_sec:.0f} seconds until the next hourly close...")
            time.sleep(sleep_sec)

# =========================================================================
# Run Bot entry
# =========================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Binance Testnet Live Trading Bot")
    parser.add_argument('--symbol', type=str, default="ETH/USDT", help="Trading Symbol (e.g. ETH/USDT, BTC/USDT)")
    parser.add_argument('--proxy', type=str, default=None, help="Proxy URL (e.g. http://127.0.0.1:1082)")
    args = parser.parse_args()

    # Retrieve Testnet Credentials from Environment variables (Best security practice)
    API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "YOUR_TESTNET_API_KEY")
    API_SECRET = os.getenv("BINANCE_TESTNET_SECRET", "YOUR_TESTNET_SECRET")
    
    if API_KEY == "YOUR_TESTNET_API_KEY":
        print("[WARNING] Please set your Binance Testnet keys in environment variables before running.")
    
    print(f"Starting bot for symbol: {args.symbol}")
    bot = BinanceTestnetBot(api_key=API_KEY, api_secret=API_SECRET, symbol=args.symbol, proxy=args.proxy)
    
    # Run loop
    bot.start_polling()

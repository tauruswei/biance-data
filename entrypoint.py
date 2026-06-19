import subprocess
import sys
import time

print("Starting both ETH and BTC trading bots...")

# Start BTC bot
btc_proc = subprocess.Popen(
    ["python", "live_trading_testnet.py", "--symbol", "BTC/USDT"],
    stdout=sys.stdout,
    stderr=sys.stderr
)

# Start ETH bot
eth_proc = subprocess.Popen(
    ["python", "live_trading_testnet.py", "--symbol", "ETH/USDT"],
    stdout=sys.stdout,
    stderr=sys.stderr
)

# Monitor both processes
try:
    while True:
        btc_code = btc_proc.poll()
        eth_code = eth_proc.poll()
        
        if btc_code is not None:
            print(f"BTC bot exited with code {btc_code}")
            sys.exit(btc_code)
            
        if eth_code is not None:
            print(f"ETH bot exited with code {eth_code}")
            sys.exit(eth_code)
            
        time.sleep(1)
except KeyboardInterrupt:
    print("Terminating bots...")
    btc_proc.terminate()
    eth_proc.terminate()
    btc_proc.wait()
    eth_proc.wait()
    print("Bots terminated.")

import subprocess
import sys
import time
import argparse
import os

parser = argparse.ArgumentParser(description="Start both ETH and BTC trading bots")
parser.add_argument('--proxy', type=str, default=None, help="Proxy URL (e.g. http://127.0.0.1:1082)")
args = parser.parse_args()

proxy_url = args.proxy or os.getenv("BINANCE_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")

print("Starting both ETH and BTC trading bots...")
if proxy_url:
    print(f"Using proxy: {proxy_url}")

btc_cmd = ["python", "live_trading_testnet.py", "--symbol", "BTC/USDT"]
eth_cmd = ["python", "live_trading_testnet.py", "--symbol", "ETH/USDT"]

if proxy_url:
    btc_cmd += ["--proxy", proxy_url]
    eth_cmd += ["--proxy", proxy_url]

# Start BTC bot
btc_proc = subprocess.Popen(
    btc_cmd,
    stdout=sys.stdout,
    stderr=sys.stderr
)

# Start ETH bot
eth_proc = subprocess.Popen(
    eth_cmd,
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

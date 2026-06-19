import urllib.request
import zipfile
import io
import pandas as pd
import sys

symbol = "BTCUSDT"
timeframe = "1h"
year = 2025
output_file = f"{symbol}_{year}_{timeframe}.csv"

all_dfs = []

print(f"开始下载 {symbol} {year} 年 {timeframe} 数据...")

for month in range(1, 13):
    month_str = f"{month:02d}"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month_str}.zip"
    print(f"正在下载 {year}-{month_str} 数据...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
            csv_filename = zip_file.namelist()[0]
            with zip_file.open(csv_filename) as f:
                # Binance vision CSV does not have headers
                df = pd.read_csv(f, header=None)
                # We need the first 6 columns: Timestamp, Open, High, Low, Close, Volume
                df = df.iloc[:, :6]
                df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                # Robustly convert timestamp to datetime
                first_ts = df['Timestamp'].iloc[0]
                ts_len = len(str(first_ts))
                if ts_len == 16:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='us')
                elif ts_len == 13:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                elif ts_len == 10:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
                else:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                
                all_dfs.append(df)
                print(f"成功导入 {year}-{month_str} 数据，共 {len(df)} 行")
    except Exception as e:
        print(f"下载/解析 {year}-{month_str} 数据失败: {e}", file=sys.stderr)
        sys.exit(1)

if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    # Ensure sorted by Timestamp and drop any duplicates
    final_df = final_df.sort_values('Timestamp').drop_duplicates(subset=['Timestamp'])
    final_df.to_csv(output_file, index=False)
    print(f"\n所有数据下载并合并成功！")
    print(f"保存路径: {output_file}")
    print(f"数据行数: {len(final_df)}")
    print(f"时间范围: {final_df['Timestamp'].min()} 至 {final_df['Timestamp'].max()}")
else:
    print("没有数据被下载", file=sys.stderr)
    sys.exit(1)

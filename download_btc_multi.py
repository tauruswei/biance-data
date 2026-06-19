import urllib.request
import urllib.error
import zipfile
import io
import pandas as pd
import sys
import os

symbol = "BTCUSDT"
timeframe = "1h"
years = list(range(2017, 2025))  # 2017 to 2024 inclusive

print(f"开始批量下载 {symbol} {timeframe} 数据，年份范围：{years[0]} - {years[-1]}...")

for year in years:
    output_file = f"{symbol}_{year}_{timeframe}.csv"
    
    # Check if file already exists. If so, we can skip downloading to save time and bandwidth
    if os.path.exists(output_file):
        print(f"\n文件 {output_file} 已存在，跳过该年份。")
        continue
        
    all_dfs = []
    print(f"\n--- 开始处理 {year} 年数据 ---")
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month_str}.zip"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
                csv_filename = zip_file.namelist()[0]
                with zip_file.open(csv_filename) as f:
                    df = pd.read_csv(f, header=None)
                    df = df.iloc[:, :6]
                    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
                    
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
                    print(f"成功导入 {year}-{month_str}，共 {len(df)} 行")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"提示: {year}-{month_str} 数据在服务器上不存在 (404)，跳过。")
            else:
                print(f"下载 {year}-{month_str} 失败 (HTTP {e.code}): {e.reason}")
        except Exception as e:
            print(f"下载/解析 {year}-{month_str} 出错: {e}")

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df = final_df.sort_values('Timestamp').drop_duplicates(subset=['Timestamp'])
        final_df.to_csv(output_file, index=False)
        print(f"成功保存 {output_file}，共 {len(final_df)} 行，时间：{final_df['Timestamp'].min()} 至 {final_df['Timestamp'].max()}")
    else:
        print(f"警告: 没有为 {year} 年成功获取任何数据。")

print("\n批量下载和处理完成！")

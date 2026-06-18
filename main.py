import ccxt
import pandas as pd
import time

# 初始化币安交易所（无需 API Key 即可获取历史公共 K 线）
exchange = ccxt.binance({
    # 解决 451 地区限制报错：如果你在受限地区（如中国大陆、美国等），需要配置代理。
    # 请取消下方注释，并将端口号（7890）替换为你自己梯子/代理软件的端口号。
    'proxies': {
        'http': 'http://127.0.0.1:1082',
        'https': 'http://127.0.0.1:1082',
    },
})

symbol = 'ETH/USDT'
timeframe = '1h'

# 2024年1月1日 0点0分0秒 的毫秒级时间戳
since = exchange.parse8601('2026-01-01T00:00:00Z')
all_klines = []

print("开始下载 2026 年 ETH 1小时线数据...")
while since < exchange.parse8601('2026-06-17T00:00:00Z'):
    # limit=1000 代表单次获取 1000 根 K 线
    klines = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
    if not klines:
        break
    since = klines[-1][0] + 60000  # 更新时间戳，取下一根线
    all_klines.extend(klines)
    time.sleep(0.1) # 略微延迟防频控

# 转换为 DataFrame 并保存为 CSV
df = pd.DataFrame(all_klines, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
df.to_csv('ETHUSDT_2026_1h.csv', index=False)
print(f"下载完成！共 {len(df)} 行数据，已保存为 ETHUSDT_2026_1h.csv")
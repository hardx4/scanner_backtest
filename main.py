import time
import requests
import pandas as pd

URL = "https://api.hyperliquid.xyz/info"

# PURR: "PURR"    KNTQ: "@334"
COIN = "@335"
ASSET_NAME = "HPL"

MAX_CANDLES_PER_REQUEST = 4999
TAXA = 0.00045  # 0.045% por lado (ajuste se quiser)
PERMITE_SHORT = True

TF_WEIGHT = {"1h": 1, "4h": 2, "1d": 3}
INTERVAL_MS = {
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def fetch_candle_page(interval, start_time, end_time):
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": COIN,
            "interval": interval,
            "startTime": int(start_time),
            "endTime": int(end_time),
        },
    }
    r = requests.post(
        URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise Exception(f"Resposta inesperada: {data}")
    return data


def get_candles(interval, start_time, end_time=None):
    if end_time is None:
        end_time = int(time.time() * 1000)

    all_rows = []
    cursor = int(start_time)

    for _ in range(50):
        if cursor >= end_time:
            break
        page_end = min(end_time, cursor + MAX_CANDLES_PER_REQUEST * INTERVAL_MS[interval])
        page = fetch_candle_page(interval, cursor, page_end)
        if not page:
            if page_end >= end_time:
                break
            cursor = page_end + 1
            continue
        all_rows.extend(page)
        last_close = int(page[-1].get("T", page[-1]["t"]))
        if last_close + 1 <= cursor:
            break
        cursor = last_close + 1
        if len(page) < MAX_CANDLES_PER_REQUEST and page_end >= end_time:
            break
        time.sleep(0.15)

    if not all_rows:
        raise Exception(f"Nenhum candle {interval}")

    df = pd.DataFrame(all_rows)
    df["open"] = pd.to_numeric(df["o"])
    df["high"] = pd.to_numeric(df["h"])
    df["low"] = pd.to_numeric(df["l"])
    df["close"] = pd.to_numeric(df["c"])
    df["volume"] = pd.to_numeric(df["v"])
    df["datetime"] = pd.to_datetime(pd.to_numeric(df["t"]), unit="ms", utc=True)
    df = df.sort_values("t").drop_duplicates(subset=["t"]).reset_index(drop=True)
    return df


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def atr(df, period=14):
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df, period=14):
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean(), plus_di, minus_di


def add_indicators(df):
    close = df["close"]
    df = df.copy()
    df["RSI"] = rsi(close)
    df["MACD"], df["SIGNAL"], df["HIST"] = macd(close)
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()
    df["ATR"] = atr(df)
    df["ADX"], df["PLUS_DI"], df["MINUS_DI"] = adx(df)
    df["ATR_PCT"] = (df["ATR"] / df["close"]) * 100
    df["VOL_MA20"] = df["volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["volume"] / df["VOL_MA20"]
    df["EMA20_SLOPE"] = ((df["EMA20"] / df["EMA20"].shift(5)) - 1) * 100
    df["EMA50_SLOPE"] = ((df["EMA50"] / df["EMA50"].shift(5)) - 1) * 100
    df["PCT_EMA200"] = ((close - df["EMA200"]) / df["EMA200"]) * 100
    df["MACD_CROSS_UP"] = (df["MACD"] > df["SIGNAL"]) & (
        df["MACD"].shift(1) <= df["SIGNAL"].shift(1)
    )
    df["MACD_CROSS_DOWN"] = (df["MACD"] < df["SIGNAL"]) & (
        df["MACD"].shift(1) >= df["SIGNAL"].shift(1)
    )
    return df


def score_row(row):
    buy = 0
    sell = 0

    if row["RSI"] > 55:
        buy += 1
    else:
        sell += 1
    if row["RSI"] > 65:
        buy += 1
    if row["RSI"] < 45:
        sell += 1
    if row["RSI"] < 35:
        sell += 1

    if row["MACD_CROSS_UP"]:
        buy += 2
    elif row["MACD"] > row["SIGNAL"]:
        buy += 1
    if row["MACD_CROSS_DOWN"]:
        sell += 2
    elif row["MACD"] < row["SIGNAL"]:
        sell += 1

    if row["close"] > row["EMA20"]:
        buy += 1
    else:
        sell += 1
    if row["EMA20"] > row["EMA50"]:
        buy += 1
    else:
        sell += 1
    if row["EMA50"] > row["EMA200"]:
        buy += 1
    else:
        sell += 1
    if row["EMA20_SLOPE"] > 0:
        buy += 1
    else:
        sell += 1
    if row["EMA50_SLOPE"] > 0:
        buy += 1
    else:
        sell += 1

    if row["VOL_RATIO"] > 1.5:
        buy += 1
    elif row["VOL_RATIO"] < 0.7:
        sell += 1
    if row["ATR_PCT"] > 15:
        sell += 1

    if row["ADX"] > 25 and row["PLUS_DI"] > row["MINUS_DI"]:
        buy += 2
    if row["ADX"] > 25 and row["MINUS_DI"] > row["PLUS_DI"]:
        sell += 2
    if row["ADX"] < 20:
        buy -= 2
        sell -= 1
    if row["PCT_EMA200"] > 35:
        buy -= 2

    return buy, sell


def last_row_at_or_before(df, ts):
    sub = df[df["t"] <= ts]
    if sub.empty:
        return None
    return sub.iloc[-1]


def run_backtest():
    end_time = int(time.time() * 1000)
    start_time = end_time - 20 * 365 * 24 * 60 * 60 * 1000

    print(f"Baixando {ASSET_NAME} ({COIN})...")
    df_1d = add_indicators(get_candles("1d", start_time, end_time))
    hist_start = int(df_1d.iloc[0]["t"])
    print(f"1d desde {df_1d.iloc[0]['datetime']} | {len(df_1d)} candles")

    df_4h = add_indicators(get_candles("4h", hist_start, end_time))
    df_1h = add_indicators(get_candles("1h", hist_start, end_time))
    print(f"4h desde {df_4h.iloc[0]['datetime']} | {len(df_4h)} candles")
    print(f"1h desde {df_1h.iloc[0]['datetime']} | {len(df_1h)} candles")

    # começa depois da EMA200 ter aquecido no 1h
    warmup = 200
    clock = df_1h.iloc[warmup:].copy()

    pos = 0  # 1 long, -1 short, 0 flat
    entry = None
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    trades = []

    for _, bar in clock.iterrows():
        ts = int(bar["t"])
        price = float(bar["close"])

        r1h = bar
        r4h = last_row_at_or_before(df_4h, ts)
        r1d = last_row_at_or_before(df_1d, ts)
        if r4h is None or r1d is None:
            continue
        if pd.isna(r1h["EMA200"]) or pd.isna(r4h["EMA200"]) or pd.isna(r1d["EMA200"]):
            continue

        buy_w = 0
        sell_w = 0
        for row, tf in ((r1h, "1h"), (r4h, "4h"), (r1d, "1d")):
            b, s = score_row(row)
            if b >= 8:
                buy_w += TF_WEIGHT[tf]
            if s >= 8:
                sell_w += TF_WEIGHT[tf]

        alvo = pos
        if buy_w >= 4:
            alvo = 1
        elif sell_w >= 4:
            alvo = -1 if PERMITE_SHORT else 0

        if alvo != pos:
            if pos != 0 and entry is not None:
                ret = (price / entry - 1) * pos
                ret -= TAXA * 2
                equity *= 1 + ret
                trades.append(
                    {
                        "saida": bar["datetime"],
                        "lado": "LONG" if pos == 1 else "SHORT",
                        "entrada": entry,
                        "saida_px": price,
                        "ret": ret,
                        "equity": equity,
                    }
                )
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / peak - 1)

            pos = alvo
            entry = price if pos != 0 else None
            if pos != 0:
                equity *= 1 - TAXA

    if pos != 0 and entry is not None:
        price = float(clock.iloc[-1]["close"])
        ret = (price / entry - 1) * pos
        ret -= TAXA
        equity *= 1 + ret
        trades.append(
            {
                "saida": clock.iloc[-1]["datetime"],
                "lado": "LONG" if pos == 1 else "SHORT",
                "entrada": entry,
                "saida_px": price,
                "ret": ret,
                "equity": equity,
            }
        )

    tdf = pd.DataFrame(trades)
    print(f"\n===== BACKTEST {ASSET_NAME} =====")
    print(f"Período 1h: {clock.iloc[0]['datetime']} → {clock.iloc[-1]['datetime']}")
    print(f"Trades: {len(tdf)}")
    if tdf.empty:
        print("Nenhum trade.")
        return

    wins = (tdf["ret"] > 0).sum()
    print(f"Win rate: {wins / len(tdf) * 100:.1f}%")
    print(f"Retorno acumulado: {(equity - 1) * 100:.2f}%")
    print(f"Melhor trade: {tdf['ret'].max() * 100:.2f}%")
    print(f"Pior trade: {tdf['ret'].min() * 100:.2f}%")
    print(f"Drawdown máximo: {max_dd * 100:.2f}%")
    print(f"Longs: {(tdf['lado'] == 'LONG').sum()} | Shorts: {(tdf['lado'] == 'SHORT').sum()}")
    print("\nÚltimos 10 trades:")
    print(
        tdf.tail(10)[["saida", "lado", "entrada", "saida_px", "ret", "equity"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    run_backtest()

import math
import time
import requests
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
URL = "https://api.hyperliquid.xyz/info"
COIN = "@272"
ASSET_NAME = "ZCASH"

WAVE_PERIOD = 34
WAVE_LOOKBACK = 8
FLAT_PCT = 0.35
STRONG_PCT = 1.20

FEE_PCT = 0.00035          # 0.035% por lado, ajuste se quiser
RISK_PCT = 0.01            # 1% da banca por trade
START_EQUITY = 10000.0
RR_TARGET = 2.0            # alvo 2R; use None para sair só por regra da onda
MAX_HOLD_BARS_1H = 72      # timeout de 72 horas

# Filtros
ALLOW_WEAK_TREND = False   # True = opera também 2-4 / baixa fraca
BLOCK_IF_DAILY_FLAT = False
BLOCK_IF_DAILY_AGAINST = True

MAX_CANDLES_PER_REQUEST = 4999
INTERVAL_MS_1H = 60 * 60 * 1000


# =========================
# DADOS
# =========================
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


def get_candles_1h(start_time, end_time=None):
    if end_time is None:
        end_time = int(time.time() * 1000)

    rows = []
    cursor = int(start_time)

    for _ in range(80):
        if cursor >= end_time:
            break

        page_end = min(end_time, cursor + MAX_CANDLES_PER_REQUEST * INTERVAL_MS_1H)
        page = fetch_candle_page("1h", cursor, page_end)

        if not page:
            if page_end >= end_time:
                break
            cursor = page_end + 1
            continue

        rows.extend(page)
        last_close = int(page[-1].get("T", page[-1]["t"]))
        nxt = last_close + 1
        if nxt <= cursor:
            break
        cursor = nxt

        if len(page) < MAX_CANDLES_PER_REQUEST and page_end >= end_time:
            break
        time.sleep(0.12)

    if not rows:
        raise Exception("Nenhum candle 1h retornado")

    df = pd.DataFrame(rows)
    df["open"] = pd.to_numeric(df["o"])
    df["high"] = pd.to_numeric(df["h"])
    df["low"] = pd.to_numeric(df["l"])
    df["close"] = pd.to_numeric(df["c"])
    df["datetime"] = pd.to_datetime(pd.to_numeric(df["t"]), unit="ms", utc=True)
    df = df.sort_values("t").drop_duplicates(subset=["t"]).reset_index(drop=True)

    # remove candle em formação
    if len(df) > 1:
        df = df.iloc[:-1].copy()
        df.reset_index(drop=True, inplace=True)
    return df


def resample_ohlc(df_1h, rule):
    ohlc = (
        df_1h.set_index("datetime")
        .resample(rule, label="right", closed="right")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        })
        .dropna()
        .reset_index()
    )
    return ohlc


# =========================
# ONDA / RELÓGIO
# =========================
def add_wave(df):
    out = df.copy()
    out["W_HIGH"] = out["high"].ewm(span=WAVE_PERIOD, adjust=False).mean()
    out["W_MID"] = out["close"].ewm(span=WAVE_PERIOD, adjust=False).mean()
    out["W_LOW"] = out["low"].ewm(span=WAVE_PERIOD, adjust=False).mean()
    mid = out["W_MID"]
    out["SLOPE"] = ((mid / mid.shift(WAVE_LOOKBACK)) - 1) * 100
    out["ANGLE"] = out["SLOPE"].apply(
        lambda x: 0.0 if pd.isna(x) else math.degrees(math.atan(x / WAVE_LOOKBACK))
    )
    return out


def clock_regime(slope):
    if pd.isna(slope):
        return "NONE"
    if abs(slope) < FLAT_PCT:
        return "FLAT"
    if slope >= STRONG_PCT:
        return "UP_STRONG"
    if slope > 0:
        return "UP_WEAK"
    if slope <= -STRONG_PCT:
        return "DOWN_STRONG"
    return "DOWN_WEAK"


def price_pos(close, w_high, w_low):
    if close > w_high:
        return "ABOVE"
    if close < w_low:
        return "BELOW"
    return "INSIDE"


def bullish(regime):
    if ALLOW_WEAK_TREND:
        return regime in ("UP_STRONG", "UP_WEAK")
    return regime == "UP_STRONG"


def bearish(regime):
    if ALLOW_WEAK_TREND:
        return regime in ("DOWN_STRONG", "DOWN_WEAK")
    return regime == "DOWN_STRONG"


# =========================
# MERGE MULTI-TF
# =========================
def build_frames(df_1h):
    h1 = add_wave(df_1h)
    h4 = add_wave(resample_ohlc(df_1h, "4h"))
    d1 = add_wave(resample_ohlc(df_1h, "1D"))

    h1["REGIME"] = h1["SLOPE"].apply(clock_regime)
    h4["REGIME"] = h4["SLOPE"].apply(clock_regime)
    d1["REGIME"] = d1["SLOPE"].apply(clock_regime)

    h1["POS"] = [
        price_pos(c, hi, lo)
        for c, hi, lo in zip(h1["close"], h1["W_HIGH"], h1["W_LOW"])
    ]

    h4_map = h4.set_index("datetime")[["REGIME", "W_HIGH", "W_LOW", "W_MID", "SLOPE"]]
    d1_map = d1.set_index("datetime")[["REGIME", "W_HIGH", "W_LOW", "W_MID", "SLOPE"]]

    # usa o último candle MAIOR já fechado
    merged = h1.copy()
    merged = pd.merge_asof(
        merged.sort_values("datetime"),
        h4_map.reset_index().rename(columns={
            "REGIME": "H4_REGIME",
            "W_HIGH": "H4_W_HIGH",
            "W_LOW": "H4_W_LOW",
            "W_MID": "H4_W_MID",
            "SLOPE": "H4_SLOPE",
        }).sort_values("datetime"),
        on="datetime",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = pd.merge_asof(
        merged.sort_values("datetime"),
        d1_map.reset_index().rename(columns={
            "REGIME": "D1_REGIME",
            "W_HIGH": "D1_W_HIGH",
            "W_LOW": "D1_W_LOW",
            "W_MID": "D1_W_MID",
            "SLOPE": "D1_SLOPE",
        }).sort_values("datetime"),
        on="datetime",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.dropna(subset=["H4_REGIME", "D1_REGIME"]).reset_index(drop=True)


# =========================
# BACKTEST
# =========================
def run_backtest(df):
    equity = START_EQUITY
    peak = START_EQUITY
    max_dd = 0.0
    position = None
    trades = []
    equity_curve = []

    for i in range(2, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        nxt = df.iloc[i + 1]

        equity_curve.append({
            "datetime": row["datetime"],
            "equity": equity,
        })

        # ---- gerencia posição aberta ----
        if position is not None:
            exit_price = None
            reason = None

            if position["side"] == "BUY":
                # stop
                if row["low"] <= position["stop"]:
                    exit_price = position["stop"]
                    reason = "stop"
                # alvo
                elif RR_TARGET is not None and row["high"] >= position["target"]:
                    exit_price = position["target"]
                    reason = "alvo"
                # quebra da onda 4h ou relógio virou
                elif row["close"] < row["H4_W_LOW"] or row["H4_REGIME"] in ("FLAT", "DOWN_WEAK", "DOWN_STRONG"):
                    exit_price = row["close"]
                    reason = "onda/relógio 4h"
                elif (i - position["entry_idx"]) >= MAX_HOLD_BARS_1H:
                    exit_price = row["close"]
                    reason = "timeout"
            else:
                if row["high"] >= position["stop"]:
                    exit_price = position["stop"]
                    reason = "stop"
                elif RR_TARGET is not None and row["low"] <= position["target"]:
                    exit_price = position["target"]
                    reason = "alvo"
                elif row["close"] > row["H4_W_HIGH"] or row["H4_REGIME"] in ("FLAT", "UP_WEAK", "UP_STRONG"):
                    exit_price = row["close"]
                    reason = "onda/relógio 4h"
                elif (i - position["entry_idx"]) >= MAX_HOLD_BARS_1H:
                    exit_price = row["close"]
                    reason = "timeout"

            if exit_price is not None:
                fee = (position["entry"] + exit_price) * FEE_PCT * position["qty"]
                if position["side"] == "BUY":
                    pnl = (exit_price - position["entry"]) * position["qty"] - fee
                else:
                    pnl = (position["entry"] - exit_price) * position["qty"] - fee

                equity += pnl
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": row["datetime"],
                    "side": position["side"],
                    "entry": position["entry"],
                    "exit": exit_price,
                    "stop": position["stop"],
                    "target": position["target"],
                    "qty": position["qty"],
                    "pnl": pnl,
                    "pnl_pct": pnl / START_EQUITY * 100,
                    "reason": reason,
                    "h4": position["h4"],
                    "d1": position["d1"],
                    "equity": equity,
                })
                position = None
            continue

        # ---- filtros de ambiente 1D ----
        if BLOCK_IF_DAILY_FLAT and row["D1_REGIME"] == "FLAT":
            continue
        if BLOCK_IF_DAILY_AGAINST and bearish(row["D1_REGIME"]) and bullish(row["H4_REGIME"]):
            continue
        if BLOCK_IF_DAILY_AGAINST and bullish(row["D1_REGIME"]) and bearish(row["H4_REGIME"]):
            continue

        # ---- setup de compra ----
        pullback_buy = (
            bullish(row["H4_REGIME"])
            and row["POS"] == "INSIDE"
            and prev["POS"] == "ABOVE"
        )
        # alternativa: já dentro da onda com 4h forte
        hold_buy = (
            bullish(row["H4_REGIME"])
            and row["POS"] == "INSIDE"
            and prev["POS"] == "INSIDE"
            and row["close"] > row["W_MID"]
            and prev["close"] <= prev["W_MID"]
        )

        # ---- setup de venda ----
        pullback_sell = (
            bearish(row["H4_REGIME"])
            and row["POS"] == "INSIDE"
            and prev["POS"] == "BELOW"
        )
        hold_sell = (
            bearish(row["H4_REGIME"])
            and row["POS"] == "INSIDE"
            and prev["POS"] == "INSIDE"
            and row["close"] < row["W_MID"]
            and prev["close"] >= prev["W_MID"]
        )

        side = None
        if pullback_buy or hold_buy:
            side = "BUY"
        elif pullback_sell or hold_sell:
            side = "SELL"

        if side is None:
            continue

        entry = float(nxt["open"])  # entra no open seguinte
        if side == "BUY":
            stop = float(min(row["W_LOW"], row["low"]))
            risk = entry - stop
            if risk <= 0:
                continue
            target = entry + risk * RR_TARGET if RR_TARGET else None
        else:
            stop = float(max(row["W_HIGH"], row["high"]))
            risk = stop - entry
            if risk <= 0:
                continue
            target = entry - risk * RR_TARGET if RR_TARGET else None

        qty = (equity * RISK_PCT) / risk
        position = {
            "side": side,
            "entry": entry,
            "stop": stop,
            "target": target,
            "qty": qty,
            "entry_time": nxt["datetime"],
            "entry_idx": i + 1,
            "h4": row["H4_REGIME"],
            "d1": row["D1_REGIME"],
        }

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(equity_curve)
    return trades_df, curve_df, equity, max_dd


def summarize(trades_df, final_equity, max_dd):
    print(f"\n===== BACKTEST {ASSET_NAME} | RELÓGIO RAGHEE =====")
    if trades_df.empty:
        print("Nenhum trade.")
        return

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    wr = len(wins) / len(trades_df) * 100
    gross_profit = wins["pnl"].sum() if len(wins) else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) else 0
    pf = (gross_profit / gross_loss) if gross_loss else np.inf
    ret = (final_equity / START_EQUITY - 1) * 100
    avg = trades_df["pnl"].mean()
    avg_w = wins["pnl"].mean() if len(wins) else 0
    avg_l = losses["pnl"].mean() if len(losses) else 0
    exp = avg / START_EQUITY * 100

    print(f"Trades: {len(trades_df)}")
    print(f"Win rate: {wr:.1f}%")
    print(f"Profit factor: {pf:.2f}")
    print(f"Retorno: {ret:.2f}%")
    print(f"Banca final: {final_equity:.2f}")
    print(f"Max drawdown: {max_dd * 100:.2f}%")
    print(f"PnL médio: {avg:.2f}")
    print(f"Médio ganho / perda: {avg_w:.2f} / {avg_l:.2f}")
    print(f"Expectativa por trade: {exp:.3f}% da banca inicial")
    print("\nPor lado:")
    print(trades_df.groupby("side")["pnl"].agg(["count", "sum", "mean"]))
    print("\nPor motivo de saída:")
    print(trades_df.groupby("reason")["pnl"].agg(["count", "sum", "mean"]))
    print("\nÚltimos 10 trades:")
    cols = ["entry_time", "exit_time", "side", "entry", "exit", "pnl", "reason", "h4", "d1"]
    print(trades_df[cols].tail(10).to_string(index=False))


if __name__ == "__main__":
    # ~2 anos de 1h. Aumente se a API deixar.
    end_time = int(time.time() * 1000)
    start_time = end_time - 800 * 24 * 60 * 60 * 1000

    print("Baixando candles 1h...")
    df_1h = get_candles_1h(start_time, end_time)
    print(f"1h candles: {len(df_1h)} | {df_1h['datetime'].iloc[0]} → {df_1h['datetime'].iloc[-1]}")

    print("Montando onda 1h / 4h / 1D...")
    df = build_frames(df_1h)

    trades_df, curve_df, final_equity, max_dd = run_backtest(df)
    summarize(trades_df, final_equity, max_dd)

    trades_df.to_csv("backtest_trades.csv", index=False)
    curve_df.to_csv("backtest_equity.csv", index=False)
    print("\nArquivos salvos: backtest_trades.csv e backtest_equity.csv")

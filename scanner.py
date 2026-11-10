#!/usr/bin/env python3
"""WBANK crypto scanner.

Scans every liquid USDT pair on Binance (public data, no API key) and reports
WHERE to enter (price level), WHEN (trigger condition) and the stop / targets.

Status meanings
  READY  the trigger fired on the last *closed* candle -> entry is live now
  WATCH  setup is forming; enter when the stated trigger happens

Not financial advice. Signals are rule-based filters, not guarantees.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

BASES = ["https://data-api.binance.vision", "https://api.binance.com"]
SKIP_SUFFIX = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "EUR", "EURI", "AEUR", "USD1", "XUSD", "BFUSD"}


# ----------------------------------------------------------------- indicators
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# --------------------------------------------------------------------- model
@dataclass
class Signal:
    symbol: str
    setup: str          # BREAKOUT | PULLBACK | REVERSAL
    status: str         # READY | WATCH
    score: int
    price: float
    entry: float
    trigger: str        # human readable "when"
    stop: float
    tp1: float
    tp2: float
    tp3: float
    risk_pct: float
    rsi: float
    vol_x: float
    trend_4h: str


def _levels(entry: float, stop: float) -> tuple[float, float, float]:
    r = entry - stop
    return entry + r, entry + 2 * r, entry + 3 * r


def htf_trend(df4h: pd.DataFrame | None) -> str:
    if df4h is None or len(df4h) < 60:
        return "n/a"
    c = df4h["close"]
    e20, e50 = ema(c, 20), ema(c, 50)
    if e20.iloc[-1] > e50.iloc[-1] and c.iloc[-1] > e50.iloc[-1] and e50.iloc[-1] > e50.iloc[-5]:
        return "UP"
    if e20.iloc[-1] < e50.iloc[-1] and c.iloc[-1] < e50.iloc[-1]:
        return "DOWN"
    return "FLAT"


def analyze(symbol: str, df: pd.DataFrame, df4h: pd.DataFrame | None = None, lookback: int = 20) -> list[Signal]:
    """df: CLOSED 1h candles only (open, high, low, close, volume)."""
    if len(df) < 80:
        return []
    c, h, l, o, v = (df[k] for k in ("close", "high", "low", "open", "volume"))
    a = atr(df).iloc[-1]
    r = rsi(c)
    e20, e50 = ema(c, 20), ema(c, 50)
    px = float(c.iloc[-1])
    if not np.isfinite(a) or a <= 0 or px <= 0:
        return []
    trend = htf_trend(df4h)
    vol_x = float(v.iloc[-1] / max(v.iloc[-lookback - 1:-1].mean(), 1e-12))
    base = dict(symbol=symbol, price=px, rsi=round(float(r.iloc[-1]), 1), vol_x=round(vol_x, 2), trend_4h=trend)
    out: list[Signal] = []

    def bonus(extra: int = 0) -> int:
        return (15 if trend == "UP" else -15 if trend == "DOWN" else 0) + extra

    def mk(setup, status, score, entry, trigger, stop):
        stop = min(stop, entry - 0.5 * a)  # never a stop tighter than half an ATR
        t1, t2, t3 = _levels(entry, stop)
        out.append(Signal(setup=setup, status=status, score=int(max(0, min(100, score))), entry=entry,
                          trigger=trigger, stop=stop, tp1=t1, tp2=t2, tp3=t3,
                          risk_pct=round((entry - stop) / entry * 100, 2), **base))

    # 1) BREAKOUT above the prior N-bar high -------------------------------
    level = float(h.iloc[-lookback - 1:-1].max())
    swing_stop = max(float(l.iloc[-5:].min()) - 0.2 * a, px - 3 * a)
    fresh = c.iloc[-3:-1].max() <= level
    if px > level and fresh and r.iloc[-1] < 80:
        s = 50 + bonus(15 if vol_x >= 2 else 8 if vol_x >= 1.5 else -10) + (5 if 55 <= r.iloc[-1] <= 72 else 0)
        mk("BREAKOUT", "READY", s, px, f"closed above {level:.6g} (breakout live)", swing_stop)
    elif px <= level and (level - px) <= min(1.0 * a, 0.015 * px) and e20.iloc[-1] > e50.iloc[-1]:
        s = 38 + bonus(5 if vol_x >= 1 else 0)
        entry = level * 1.001
        mk("BREAKOUT", "WATCH", s, entry, f"enter on a 1h close above {level:.6g}", max(float(l.iloc[-5:].min()) - 0.2 * a, entry - 3 * a))

    # 2) PULLBACK to EMA20/50 inside an uptrend ----------------------------
    uptrend = e20.iloc[-1] > e50.iloc[-1] and e50.iloc[-1] > e50.iloc[-6]
    if uptrend:
        touched = l.iloc[-1] <= e20.iloc[-1] * 1.002 and px > e50.iloc[-1]
        bullish = px > o.iloc[-1] and px > c.iloc[-2]
        if touched and bullish and 38 <= r.iloc[-1] <= 60 and r.iloc[-1] > r.iloc[-2]:
            s = 45 + bonus(10 if vol_x >= 1.2 else 0) + (5 if r.iloc[-1] < 50 else 0)
            mk("PULLBACK", "READY", s, px, "bounce confirmed off EMA20 (bullish close)", float(min(l.iloc[-3:].min(), e50.iloc[-1])) - 0.3 * a)
        elif not touched and 0 < (px - e20.iloc[-1]) <= 1.5 * a and r.iloc[-1] < 58 and px > e50.iloc[-1]:
            entry = float(e20.iloc[-1])
            s = 35 + bonus(0)
            mk("PULLBACK", "WATCH", s, entry, f"limit buy zone {entry:.6g} (EMA20); needs bullish candle there", float(e50.iloc[-1]) - 0.3 * a)

    # 3) OVERSOLD REVERSAL (counter-trend, lower base score) ---------------
    if r.iloc[-6:-1].min() < 30 and r.iloc[-1] > r.iloc[-2] and px > h.iloc[-2] and px > o.iloc[-1]:
        s = 35 + bonus(8 if vol_x >= 1.5 else 0)
        mk("REVERSAL", "READY", s, px, "RSI<30 flush + close above prior high", float(l.iloc[-5:].min()) - 0.2 * a)

    return out


# --------------------------------------------------------------------- data
class Binance:
    def __init__(self, base: str | None = None):
        import requests
        self.s = requests.Session()
        self.bases = [base] if base else BASES
        self.base = None

    def _get(self, path: str, **params):
        last = None
        for b in ([self.base] if self.base else self.bases):
            try:
                r = self.s.get(b + path, params=params, timeout=15)
                if r.status_code == 200:
                    self.base = b
                    return r.json()
                last = f"{b}{path} -> HTTP {r.status_code}"
            except Exception as e:  # network error: try next base
                last = f"{b}{path} -> {e}"
        raise RuntimeError(last)

    def symbols(self, min_vol: float, top: int) -> list[str]:
        rows = self._get("/api/v3/ticker/24hr")
        rows = [x for x in rows if x["symbol"].endswith("USDT") and not x["symbol"].endswith(SKIP_SUFFIX)
                and x["symbol"][:-4] not in STABLES and float(x["quoteVolume"]) >= min_vol]
        rows.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        return [x["symbol"] for x in rows[:top]]

    def klines(self, symbol: str, interval: str, limit: int = 250) -> pd.DataFrame:
        raw = self._get("/api/v3/klines", symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(raw).iloc[:, :7]
        df.columns = ["t", "open", "high", "low", "close", "volume", "close_t"]
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df[df["close_t"] < time.time() * 1000].reset_index(drop=True)  # closed candles only


def scan(client, symbols: list[str], tf: str, workers: int = 8) -> list[Signal]:
    def one(sym):
        try:
            return analyze(sym, client.klines(sym, tf), client.klines(sym, "4h"))
        except Exception as e:
            print(f"  ! {sym}: {e}", file=sys.stderr)
            return []
    with ThreadPoolExecutor(workers) as ex:
        res = [s for lst in ex.map(one, symbols) for s in lst]
    return sorted(res, key=lambda s: (s.status != "READY", -s.score))


# ------------------------------------------------------------------- output
def fmt(x: float) -> str:
    return f"{x:.6g}"


def print_table(sigs: list[Signal]) -> None:
    if not sigs:
        print("No setups right now.")
        return
    print(f"{'COIN':<11}{'SETUP':<9}{'STATUS':<6}{'SCR':>4} {'ENTRY':>11} {'STOP':>11} {'TP1':>11} {'TP2':>11} {'TP3':>11} {'RISK%':>6} {'RSI':>5} {'VOLx':>5} 4H")
    for s in sigs:
        print(f"{s.symbol:<11}{s.setup:<9}{s.status:<6}{s.score:>4} {fmt(s.entry):>11} {fmt(s.stop):>11} {fmt(s.tp1):>11} {fmt(s.tp2):>11} {fmt(s.tp3):>11} {s.risk_pct:>6} {s.rsi:>5} {s.vol_x:>5} {s.trend_4h}")
        print(f"{'':<11}-> WHEN: {s.trigger}")


def notify(sigs: list[Signal]) -> None:
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (tok and chat and sigs):
        return
    import requests
    for s in sigs:
        msg = (f"{s.status} {s.setup} {s.symbol} (score {s.score})\nWhen: {s.trigger}\nEntry {fmt(s.entry)} | Stop {fmt(s.stop)}\n"
               f"TP1 {fmt(s.tp1)} TP2 {fmt(s.tp2)} TP3 {fmt(s.tp3)} | risk {s.risk_pct}%")
        try:
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": msg}, timeout=10)
        except Exception as e:
            print(f"telegram failed: {e}", file=sys.stderr)


def demo_client():
    """Offline random-walk market for trying the pipeline without internet."""
    rng = np.random.default_rng(7)

    class Demo:
        def symbols(self, *_):
            return [f"COIN{i}USDT" for i in range(40)]

        def klines(self, sym, interval, limit=250):
            n = 250
            ret = rng.normal(0.0004, 0.01, n)
            c = 100 * np.exp(np.cumsum(ret))
            o = np.r_[c[0], c[:-1]]
            hi, lo = np.maximum(o, c) * (1 + abs(rng.normal(0, .004, n))), np.minimum(o, c) * (1 - abs(rng.normal(0, .004, n)))
            return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": rng.uniform(80, 120, n)})
    return Demo()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tf", default="1h", help="signal timeframe (15m, 1h, 4h...) default 1h")
    p.add_argument("--top", type=int, default=150, help="scan N most liquid USDT pairs")
    p.add_argument("--min-vol", type=float, default=5_000_000, help="min 24h quote volume in USDT")
    p.add_argument("--min-score", type=int, default=50)
    p.add_argument("--watch", type=int, default=0, metavar="SEC", help="rescan every SEC seconds, alert on new signals")
    p.add_argument("--json", help="also write results to this JSON file")
    p.add_argument("--base", help="API base URL (e.g. https://api.binance.us for US users)")
    p.add_argument("--demo", action="store_true", help="offline synthetic data")
    a = p.parse_args()

    client = demo_client() if a.demo else Binance(a.base)
    seen: set[tuple] = set()
    while True:
        syms = client.symbols(a.min_vol, a.top)
        print(f"\n[{time.strftime('%H:%M:%S')}] scanning {len(syms)} coins on {a.tf} ...")
        sigs = [s for s in scan(client, syms, a.tf) if s.score >= a.min_score]
        print_table(sigs)
        if a.json:
            with open(a.json, "w") as f:
                json.dump([asdict(s) for s in sigs], f, indent=2)
        new = [s for s in sigs if (s.symbol, s.setup, s.status) not in seen]
        seen.update((s.symbol, s.setup, s.status) for s in new)
        if a.watch:
            notify(new)
            time.sleep(a.watch)
        else:
            break


if __name__ == "__main__":
    main()

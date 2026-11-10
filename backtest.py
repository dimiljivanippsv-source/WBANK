#!/usr/bin/env python3
"""Walk-forward backtest of scanner.py READY signals on real Binance candles.

Realism rules: entry at NEXT candle open, fees+slippage charged, stop wins if
stop and target are hit in the same candle, no leverage, 1% risk per trade.
Parameters are compared in-sample (first 2/3) vs out-of-sample (last 1/3).
"""
from __future__ import annotations
import argparse, itertools, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
from scanner import Binance, analyze

DATA = os.path.join(os.path.dirname(__file__), "data")
COST = 0.0020  # round-trip fees + slippage as fraction of price (0.1% fee x2, ~0 slippage added in R terms below)


def download(client: Binance, sym: str, bars: int) -> pd.DataFrame:
    os.makedirs(DATA, exist_ok=True)
    path = f"{DATA}/{sym}_1h_{bars}.pkl"
    if os.path.exists(path):
        return pd.read_pickle(path)
    end, rows = int(time.time() * 1000), []
    while len(rows) < bars:
        raw = client._get("/api/v3/klines", symbol=sym, interval="1h", limit=1000, endTime=end)
        if not raw:
            break
        rows = raw + rows
        end = raw[0][0] - 1
    df = pd.DataFrame(rows).iloc[-bars:, :7]
    df.columns = ["t", "open", "high", "low", "close", "volume", "close_t"]
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[df["close_t"] < time.time() * 1000].reset_index(drop=True)
    df.to_pickle(path)
    return df


def to4h(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index(pd.to_datetime(df["t"], unit="ms"))
    g = d.resample("4h", origin="epoch").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "close_t": "last", "t": "first"})
    g["n"] = d["close"].resample("4h", origin="epoch").count()
    return g[g["n"] == 4].reset_index(drop=True)  # complete 4h bars only


def gen_signals(args):
    sym, df = args
    h4 = to4h(df)
    h4_close = h4["close_t"].to_numpy()
    out = []
    for i in range(120, len(df) - 2):
        k = np.searchsorted(h4_close, df["close_t"].iloc[i], side="right")  # 4h bars fully closed by now
        for s in analyze(sym, df.iloc[max(0, i - 249): i + 1].reset_index(drop=True), h4.iloc[max(0, k - 120): k]):
            if s.status == "READY":
                out.append(dict(sym=sym, i=i, t=int(df["close_t"].iloc[i]), setup=s.setup, score=s.score, stop=s.stop, trend=s.trend_4h))
    return sym, out


def simulate(arr, sg: dict, tp_r: float, hold: int):
    """arr = dict of numpy arrays for the symbol. Returns (entry_t, exit_t, net_ret, risk_frac) or None."""
    j = sg["i"] + 1
    entry = arr["open"][j]
    risk = entry - sg["stop"]
    if risk <= 0 or risk / entry > 0.08:
        return None
    tp = entry + tp_r * risk
    last = min(j + hold, len(arr["close"]) - 1)
    exit_p, exit_j = None, last
    for k in range(j, last + 1):
        if arr["low"][k] <= sg["stop"]:
            exit_p, exit_j = min(arr["open"][k], sg["stop"]), k; break
        if arr["high"][k] >= tp:
            exit_p, exit_j = tp, k; break
    if exit_p is None:
        exit_p = arr["close"][exit_j]
    return int(arr["t"][j]), int(arr["close_t"][exit_j]), (exit_p - entry) / entry - COST, risk / entry


def portfolio(trades, risk_frac=0.01, max_open=5, start=1.0):
    """trades: list of (entry_t, exit_t, ret, risk_pct, sym). Compounding, no leverage."""
    eq, open_pos, pnl_events = start, [], []
    for et, xt, ret, rp, sym in sorted(trades):
        for p in [p for p in open_pos if p[0] <= et]:  # realise closed ones first
            open_pos.remove(p); eq += p[1]
            pnl_events.append((p[0], eq))
        if len(open_pos) >= max_open or any(p[2] == sym for p in open_pos):
            continue
        notional = min(eq * risk_frac / rp, eq / max_open)
        open_pos.append((xt, notional * ret, sym))
    for p in sorted(open_pos):
        eq += p[1]; pnl_events.append((p[0], eq))
    return start, pnl_events


def stats(start, ev, n_days):
    if not ev:
        return dict(trades=0, tot=0, daily=0, dd=0, hit2=0, worst=0, best=0)
    s = pd.Series([e for _, e in ev], index=pd.to_datetime([t for t, _ in ev], unit="ms"))
    daily = s.resample("1D").last().ffill()
    daily = pd.concat([pd.Series([start]), daily]).pct_change().dropna()
    dd = (s / s.cummax() - 1).min()
    return dict(trades=len(ev), tot=s.iloc[-1] / start - 1, daily=daily.mean(), hit2=(daily >= 0.02).mean(), dd=dd,
                worst=daily.min(), best=daily.max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coins", type=int, default=40)
    p.add_argument("--bars", type=int, default=8760, help="1h bars of history (8760 = 1 year)")
    a = p.parse_args()
    client = Binance("https://data-api.binance.vision")
    syms = client.symbols(20_000_000, a.coins)
    print(f"downloading {len(syms)} coins x {a.bars} 1h bars ...", flush=True)
    data = {s: download(client, s, a.bars) for s in syms}
    data = {s: d for s, d in data.items() if len(d) > 2000}
    cache = f"{DATA}/signals_{len(data)}_{a.bars}.pkl"
    if os.path.exists(cache):
        sigs = pickle.load(open(cache, "rb"))
    else:
        print(f"generating signals on {len(data)} coins (slow, parallel) ...", flush=True)
        with Pool() as pool:
            sigs = dict(pool.map(gen_signals, list(data.items())))
        pickle.dump(sigs, open(cache, "wb"))
    allsig = [s for v in sigs.values() for s in v]
    print(f"{len(allsig)} READY signals total")
    t0 = min(d["t"].iloc[0] for d in data.values()); t1 = max(d["close_t"].iloc[-1] for d in data.values())
    split = t0 + (t1 - t0) * 2 // 3
    days = (t1 - t0) / 86_400_000
    # buy & hold benchmark
    bh = np.mean([d["close"].iloc[-1] / d["close"].iloc[0] - 1 for d in data.values()])
    print(f"period {days:.0f} days; equal-weight buy&hold of these coins: {bh:+.1%}\n")

    arrs = {k: {c: d[c].to_numpy() for c in ("open", "high", "low", "close", "t", "close_t")} for k, d in data.items()}
    sim = {(tpr, hold): [simulate(arrs[x["sym"]], x, tpr, hold) for x in allsig] for tpr in (1, 2, 3) for hold in (24, 72)}
    print("simulated exits", flush=True)
    grid = itertools.product(["ALL", "BREAKOUT", "PULLBACK", "REVERSAL"], [50, 60, 70], [1, 2, 3], [24, 72], [False, True])
    rows = []
    for setup, minsc, tpr, hold, uponly in grid:
        tr = [(*r, s["sym"]) for s, r in zip(allsig, sim[(tpr, hold)])
              if r and (setup == "ALL" or s["setup"] == setup) and s["score"] >= minsc and (not uponly or s["trend"] == "UP")]
        for name, sub in (("IS", [x for x in tr if x[0] < split]), ("OOS", [x for x in tr if x[0] >= split])):
            st0, ev = portfolio(sub)
            nd = (split - t0) / 86_400_000 if name == "IS" else (t1 - split) / 86_400_000
            rows.append(dict(setup=setup, minsc=minsc, tp_r=tpr, hold=hold, uptrend_only=uponly, part=name, wr=np.mean([x[2] > 0 for x in sub]) if sub else 0, **stats(st0, ev, nd)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{DATA}/results.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    piv = df.pivot_table(index=["setup", "minsc", "tp_r", "hold", "uptrend_only"], columns="part", values=["daily", "tot", "dd", "trades", "wr"]).reset_index()
    piv.columns = ["_".join(c).strip("_") for c in piv.columns]
    top = piv[piv["trades_IS"] >= 30].sort_values("daily_IS", ascending=False).head(12)
    print("TOP 12 configs by IN-SAMPLE avg daily return (then how they did OUT-of-sample):")
    print(top[["setup", "minsc", "tp_r", "hold", "uptrend_only", "trades_IS", "wr_IS", "daily_IS", "dd_IS", "trades_OOS", "wr_OOS", "daily_OOS", "tot_OOS", "dd_OOS"]].to_string(index=False))
    full = df[df.part == "OOS"]
    print(f"\nBest OOS avg daily return of ANY config: {full.daily.max():+.3%} ; configs with OOS avg >= 2%/day: {(full.daily >= 0.02).sum()} of {len(full)}")
    print(f"Configs profitable OOS: {(full.tot > 0).sum()} of {len(full)}")


if __name__ == "__main__":
    main()

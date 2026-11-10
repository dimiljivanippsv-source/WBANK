import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scanner import analyze


def frame(closes, vols=None):
    c = np.array(closes, float)
    o = np.r_[c[0], c[:-1]]
    hi, lo = np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c,
                         "volume": np.array(vols if vols is not None else [100] * len(c), float)})


def uptrend(n=240, drift=0.002):
    rng = np.random.default_rng(1)
    return list(100 * np.exp(np.cumsum(rng.normal(drift, 0.004, n))))


def test_breakout_ready():
    c = uptrend()
    top = max(c[-21:]) 
    c = c[:-1] + [top * 1.012]
    v = [100] * (len(c) - 1) + [300]
    sigs = analyze("TESTUSDT", frame(c, v), frame(uptrend()))
    b = [s for s in sigs if s.setup == "BREAKOUT" and s.status == "READY"]
    assert b, sigs
    s = b[0]
    assert s.stop < s.entry < s.tp1 < s.tp2 < s.tp3 and s.score >= 60 and s.trend_4h == "UP"


def test_breakout_watch_when_just_below():
    c = uptrend()
    level_hi = frame(c)["high"].iloc[-21:-1].max()
    c = c[:-1] + [level_hi * 0.995]
    sigs = analyze("TESTUSDT", frame(c), frame(uptrend()))
    assert any(s.setup == "BREAKOUT" and s.status == "WATCH" for s in sigs), sigs


def test_flat_market_no_ready():
    c = list(100 + np.sin(np.arange(240) / 5) * 0.01)
    assert not [s for s in analyze("FLATUSDT", frame(c)) if s.status == "READY"]


def test_too_short():
    assert analyze("X", frame([1] * 30)) == []


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)

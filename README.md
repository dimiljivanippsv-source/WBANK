# WBANK Crypto Scanner

Scans the most liquid USDT pairs on Binance (public API, no key) and tells you
**where** to enter, **when** the entry triggers, and the **stop / TP1-TP3**.

```bash
pip install -r requirements.txt
python scanner.py                       # one scan, 1h signals, top 150 coins
python scanner.py --tf 15m --top 300    # faster timeframe, more coins
python scanner.py --watch 300           # rescan every 5 min, alert on NEW setups
python scanner.py --json out.json       # save results
python scanner.py --demo                # offline test with fake data
python tests/test_scanner.py            # logic tests
```

US users: Binance.com is geo-blocked, use `--base https://api.binance.us`.
Telegram alerts (with `--watch`): set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

## Setups
| Setup | READY (enter now) | WATCH (enter when...) |
|---|---|---|
| BREAKOUT | candle closed above the 20-bar high, volume above average, RSI < 80 | price within 1 ATR below the level; enter on a close above it |
| PULLBACK | uptrend (EMA20 > EMA50), bullish bounce off EMA20, RSI 38-60 and rising | price drifting down to EMA20; limit-buy zone shown |
| REVERSAL | RSI < 30 flush then a close above the prior high (counter-trend) | - |

Score (0-100) adds a bonus when the 4h trend is UP, a penalty when DOWN, and
rewards volume expansion. Stops are ATR/structure based; TPs are 1R/2R/3R.
Only closed candles are used, so signals do not repaint.

Rule-based screening, not financial advice. Backtest and size positions before risking money.

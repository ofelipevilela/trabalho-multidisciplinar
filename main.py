# main.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# local modules (root-level)
from model_heston import estimate_heston_vol
from model_garch import estimate_garch_vol
from signals import SignalConfig, build_signals


# ========= USER CONFIG (edit here) =========
TICKER = "^GSPC"         # e.g., "^GSPC", "NVDA", "AAPL", "AMD"
START  = "2010-01-01"    # start date for historical data
PROFILE = "aggressive"     # "conservative" | "moderate" | "aggressive"
# ==========================================


def load_prices_yf(ticker: str, start: str = "2005-01-01") -> pd.Series:
    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError(
            "yfinance is required. Install it with:\n"
            "  python -m pip install yfinance"
        ) from e

    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    df.index.name = "Date"

    # prefer Adj Close; fallback to Close
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].copy()
    s.name = "Adj Close"   # set the Series name directly (no .rename("..."))
    return s

def _strategy_returns(sig: pd.DataFrame) -> pd.Series:
    """Daily strategy returns using long-only position with 1-day lag on execution."""
    asset_ret = sig["returns"].fillna(0.0)
    pos = sig["position"].shift(1).fillna(0).astype(float)
    strat_ret = (asset_ret * pos).rename("strategy_ret")
    return strat_ret


def _equity_curve(strat_ret: pd.Series) -> pd.Series:
    """Cumulative equity (starting at 1.0)."""
    return (1.0 + strat_ret).cumprod().rename("equity")


def _drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown series from equity."""
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return dd.rename("drawdown")


def _rolling_sharpe(returns: pd.Series, window: int = 63, periods_per_year: int = 252) -> pd.Series:
    """Rolling Sharpe (no risk-free): mean/std over window, annualized."""
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std().replace(0, np.nan)
    rs = (mu / sd) * np.sqrt(periods_per_year)
    return rs.rename(f"rolling_sharpe_{window}")


def _hit_rate(strat_ret: pd.Series, asset_ret: pd.Series) -> float:
    """Directional hit rate on days with position != 0."""
    mask = strat_ret != 0
    if mask.sum() == 0:
        return float("nan")
    correct = np.sign(strat_ret[mask]) == np.sign(asset_ret[mask])
    return float(correct.mean())


def visualize(sig: pd.DataFrame, cfg, ticker: str) -> None:
    """
    Display charts and performance metrics interactively (no files saved).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # --- metrics prep ---
    strat_ret = _strategy_returns(sig)
    equity = _equity_curve(strat_ret)
    dd = _drawdown(equity)

    total_return = equity.iloc[-1] - 1.0
    cagr = (equity.iloc[-1] ** (252 / max(1, len(sig)))) - 1.0
    vol_ann = sig["returns"].std() * np.sqrt(252)
    strat_vol_ann = strat_ret.std() * np.sqrt(252)
    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() > 0 else np.nan
    hit = _hit_rate(strat_ret, sig["returns"])

    print("\n=== STRATEGY METRICS ===")
    print(f"Ticker             : {ticker}")
    print(f"Profile            : {cfg.profile}")
    print(f"Trades (entries)   : {int((sig['position'].diff() == 1).sum())}")
    print(f"Total Return       : {total_return: .2%}")
    print(f"CAGR (approx)      : {cagr: .2%}")
    print(f"Asset Vol (ann)    : {vol_ann: .2%}")
    print(f"Strategy Vol (ann) : {strat_vol_ann: .2%}")
    print(f"Sharpe (no RF)     : {sharpe: .2f}")
    print(f"Hit Rate           : {hit: .2%}")

    # 1️⃣ PRICE + EMAs + POSITION
    plt.figure(figsize=(10, 5))
    plt.plot(sig.index, sig["price"], label="Price", lw=1.2)
    plt.plot(sig.index, sig[f"ema{cfg.ema_fast}"], label=f"EMA {cfg.ema_fast}")
    plt.plot(sig.index, sig[f"ema{cfg.ema_slow}"], label=f"EMA {cfg.ema_slow}")
    plt.fill_between(sig.index, sig["price"].min(), sig["price"].max(),
                     where=sig["position"].astype(bool), color="green", alpha=0.1, label="Position=1")
    plt.title(f"{ticker} — Price & EMAs (position shaded)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2️⃣ VOLATILITIES
    plt.figure(figsize=(10, 4))
    plt.plot(sig.index, sig["garch_vol"], label="GARCH Vol")
    plt.plot(sig.index, sig["heston_vol"], label="Heston Vol")
    plt.plot(sig.index, sig["vol_pred_cons"], label="Consensus Vol")
    vol_hist_col = [c for c in sig.columns if c.startswith("vol_hist_")][0]
    plt.plot(sig.index, sig[vol_hist_col], label="Realized Vol (7d)")
    plt.title(f"{ticker} — Predicted vs Realized Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3️⃣ Z-SCORE
    plt.figure(figsize=(10, 4))
    plt.plot(sig.index, sig["zscore"], label="Z-Score")
    plt.axhline(cfg.z_thresholds[cfg.profile], color="r", linestyle="--", label="Threshold")
    plt.title(f"{ticker} — Z-Score ({cfg.profile} profile)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4️⃣ EQUITY & DRAWDOWN
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(sig.index, equity, color="blue")
    ax[0].set_title("Strategy Equity")
    ax[1].plot(sig.index, dd, color="red")
    ax[1].set_title("Drawdown")
    plt.tight_layout()
    plt.show()

    # 5️⃣ HISTOGRAM
    plt.figure(figsize=(7, 4))
    plt.hist(strat_ret.dropna(), bins=40, alpha=0.7)
    plt.title(f"{ticker} — Strategy Daily Returns Distribution")
    plt.tight_layout()
    plt.show()

def run() -> None:
    # 1) Load prices from Yahoo Finance
    prices = load_prices_yf(TICKER, start=START)

    # 2) Returns
    returns = prices.pct_change().dropna()

    # 3) Volatility estimates (daily, same index/scale)
    heston_vol = estimate_heston_vol(returns, window=7)
    garch_vol  = estimate_garch_vol(returns, variant="GARCH")

    # Align indices
    idx = prices.index.intersection(heston_vol.index).intersection(garch_vol.index)
    prices     = prices.reindex(idx)
    heston_vol = heston_vol.reindex(idx)
    garch_vol  = garch_vol.reindex(idx)

    # 4) Build signals (Slides 4→9)
    cfg = SignalConfig(profile=PROFILE)
    sig = build_signals(prices, garch_vol, heston_vol, cfg=cfg)

    # 5) Save output
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True, parents=True)
    out_csv = outdir / "signals.csv"
    sig.to_csv(out_csv, index=True)

    # 6) Quick summary
    print("\n=== Meta-Strategy Signals ===")
    print(f"Source     : Ticker: {TICKER} (start={START})")
    print(f"Profile    : {PROFILE}")
    print(f"Rows       : {len(sig)}")
    print(f"Output CSV : {out_csv}")
    print("\nTail:")
    cols = ["price", "garch_vol", "heston_vol", "zscore", "risk_state", "agree_flag", "position"]
    print(sig[cols].tail(10))

    #7) Visualization & metrics
    visualize(sig, cfg, TICKER)
    
if __name__ == "__main__":
    run()

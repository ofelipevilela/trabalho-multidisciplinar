
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import load_prices_yf
from model_heston import estimate_heston_vol
from model_garch import estimate_garch_vol
from signals import SignalConfig, build_signals

# Config
TICKER = "^GSPC"
START = "2018-01-01"
PROFILE = "moderate"

def analyze():
    print(f"Analyzing {TICKER} from {START}...")
    
    # 1. Load Data
    prices = load_prices_yf(TICKER, start=START)
    returns = prices.pct_change().dropna()
    
    # 2. Estimate Vols
    print("Estimating Heston...")
    heston_vol = estimate_heston_vol(returns, ticker=TICKER, verbose=False)
    print("Estimating GARCH...")
    garch_vol = estimate_garch_vol(returns, variant="GARCH")
    
    # 3. Align
    idx = prices.index.intersection(heston_vol.index).intersection(garch_vol.index)
    prices = prices.reindex(idx)
    heston_vol = heston_vol.reindex(idx)
    garch_vol = garch_vol.reindex(idx)
    
    # 4. Build Signals (Predicted)
    cfg = SignalConfig(profile=PROFILE)
    sig_pred = build_signals(prices, garch_vol, heston_vol, cfg=cfg)
    
    # 5. Build Signals (Historical)
    vol_hist_col = [c for c in sig_pred.columns if c.startswith("vol_hist_")][0]
    vol_hist_series = sig_pred[vol_hist_col]
    sig_hist = build_signals(prices, vol_hist_series, vol_hist_series, cfg=cfg)
    
    # 6. Compare Statistics
    print("\n--- Volatility Statistics ---")
    vol_pred = sig_pred["vol_pred_cons"]
    vol_hist = sig_hist["vol_pred_cons"] # This is actually the historical vol passed in
    
    print(f"Mean Vol (Pred): {vol_pred.mean():.6f}")
    print(f"Mean Vol (Hist): {vol_hist.mean():.6f}")
    print(f"Std Vol (Pred):  {vol_pred.std():.6f}")
    print(f"Std Vol (Hist):  {vol_hist.std():.6f}")
    
    # Noise / Stability (Mean Absolute Change)
    print(f"Mean Abs Change (Pred): {vol_pred.diff().abs().mean():.6f}")
    print(f"Mean Abs Change (Hist): {vol_hist.diff().abs().mean():.6f}")
    
    print("\n--- Z-Score Statistics ---")
    z_pred = sig_pred["zscore"]
    z_hist = sig_hist["zscore"]
    
    print(f"Mean Z (Pred): {z_pred.mean():.4f}")
    print(f"Mean Z (Hist): {z_hist.mean():.4f}")
    print(f"Std Z (Pred):  {z_pred.std():.4f}")
    print(f"Std Z (Hist):  {z_hist.std():.4f}")
    
    # Threshold Crossings
    z_thresh = cfg.z_thresholds[PROFILE]
    buy_thresh = z_thresh["buy"]
    sell_thresh = z_thresh["sell"]
    
    print(f"\nThresholds: Buy={buy_thresh}, Sell={sell_thresh}")
    
    pred_buy_zone = (z_pred < buy_thresh).sum() / len(z_pred)
    pred_sell_zone = (z_pred > sell_thresh).sum() / len(z_pred)
    
    hist_buy_zone = (z_hist < buy_thresh).sum() / len(z_hist)
    hist_sell_zone = (z_hist > sell_thresh).sum() / len(z_hist)
    
    print(f"Time in Buy Zone (Pred):  {pred_buy_zone:.1%}")
    print(f"Time in Buy Zone (Hist):  {hist_buy_zone:.1%}")
    print(f"Time in Sell Zone (Pred): {pred_sell_zone:.1%}")
    print(f"Time in Sell Zone (Hist): {hist_sell_zone:.1%}")

    # Signal Stability
    print("\n--- Signal Stability ---")
    pred_switches = sig_pred["position"].diff().abs().sum()
    hist_switches = sig_hist["position"].diff().abs().sum()
    
    print(f"Position Changes (Pred): {int(pred_switches)}")
    print(f"Position Changes (Hist): {int(hist_switches)}")
    
    # Correlation with Future Returns?
    # Ideally, low Z-score (Calmaria) should predict positive returns
    # High Z-score (Risco) should predict negative/volatile returns
    
    next_ret = sig_pred["returns"].shift(-1)
    
    corr_pred = z_pred.corr(next_ret)
    corr_hist = z_hist.corr(next_ret)
    
    print(f"\nZ-Score Corr with Next Day Return (Pred): {corr_pred:.4f}")
    print(f"Z-Score Corr with Next Day Return (Hist): {corr_hist:.4f}")

if __name__ == "__main__":
    analyze()

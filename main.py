# main.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# local modules (root-level)
from model_heston import estimate_heston_vol
from model_garch import estimate_garch_vol
from signals import SignalConfig, build_signals, build_ema_only_signals


# ========= USER CONFIG (edit here) =========
TICKER = "^GSPC"         # e.g., "^GSPC", "NVDA", "AAPL", "AMD"
START  = "2018-01-01"    # start date for historical data
PROFILE = "moderate"     # "conservative" | "moderate" | "aggressive"
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
    
    # Handle MultiIndex columns (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex columns
        df.columns = df.columns.get_level_values(0)
    
    df.index.name = "Date"

    # prefer Adj Close; fallback to Close
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].copy()
    s.name = "Adj Close"   # set the Series name directly (no .rename("..."))
    
    # Ensure index is a simple DatetimeIndex
    if isinstance(s.index, pd.MultiIndex):
        s.index = s.index.get_level_values(0)
    
    return s

def _strategy_returns(sig: pd.DataFrame) -> pd.Series:
    """Daily strategy returns using long/short positions with 1-day lag on execution."""
    asset_ret = sig["returns"].fillna(0.0)
    pos = sig["position"].shift(1).fillna(0).astype(float)
    # Position: +1 (compra), -1 (venda), 0 (neutro)
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

    # Count trades
    buy_entries = int((sig['position'].diff() == 1).sum())
    sell_entries = int((sig['position'].diff() == -1).sum())
    total_trades = buy_entries + sell_entries
    
    # Position statistics
    days_long = int((sig['position'] == 1).sum())
    days_short = int((sig['position'] == -1).sum())
    days_neutral = int((sig['position'] == 0).sum())
    
    print("\n=== STRATEGY METRICS ===")
    print(f"Ticker             : {ticker}")
    print(f"Profile            : {cfg.profile}")
    print(f"Total Trades       : {total_trades} (Compras: {buy_entries}, Vendas: {sell_entries})")
    print(f"Days in Position   : Long={days_long}, Short={days_short}, Neutral={days_neutral}")
    print(f"Total Return       : {total_return: .2%}")
    print(f"CAGR (approx)      : {cagr: .2%}")
    print(f"Asset Vol (ann)    : {vol_ann: .2%}")
    print(f"Strategy Vol (ann) : {strat_vol_ann: .2%}")
    print(f"Sharpe (no RF)     : {sharpe: .2f}")
    print(f"Hit Rate           : {hit: .2%}")

    # 1️⃣ PRICE + EMAs + POSITION (com marcadores de entrada/saída)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot price and EMAs
    ax.plot(sig.index, sig["price"], label="Price", lw=1.5, color="black", zorder=5)
    ax.plot(sig.index, sig[f"ema{cfg.ema_fast}"], label=f"EMA {cfg.ema_fast}", lw=1.2, alpha=0.8)
    ax.plot(sig.index, sig[f"ema{cfg.ema_slow}"], label=f"EMA {cfg.ema_slow}", lw=1.2, alpha=0.8)
    
    # Shade long positions (green) and short positions (red)
    long_mask = sig["position"] == 1
    short_mask = sig["position"] == -1
    ax.fill_between(sig.index, sig["price"].min(), sig["price"].max(),
                    where=long_mask, color="green", alpha=0.12, label="Long Position", zorder=1)
    ax.fill_between(sig.index, sig["price"].min(), sig["price"].max(),
                    where=short_mask, color="red", alpha=0.12, label="Short Position", zorder=1)
    
    # Detect entry and exit points
    position_diff = sig["position"].diff()
    prev_position = sig["position"].shift(1).fillna(0).astype(int)
    
    # Buy entries: position changes TO 1 (from 0 or -1)
    # Only count if previous position was not 1
    buy_entries_mask = (position_diff == 1) & (prev_position != 1)
    buy_entries_idx = buy_entries_mask[buy_entries_mask].index
    if len(buy_entries_idx) > 0:
        buy_entry_prices = sig.loc[buy_entries_idx, "price"]
        ax.scatter(buy_entries_idx, buy_entry_prices, marker="^", color="lime", 
                  s=120, zorder=15, label=f"Buy Entry ({len(buy_entries_idx)})", 
                  edgecolors="darkgreen", linewidths=2, alpha=0.9)
    
    # Sell entries: position changes TO -1 (from 0 or 1)
    # Only count if previous position was not -1
    sell_entries_mask = (position_diff == -1) & (prev_position != -1)
    sell_entries_idx = sell_entries_mask[sell_entries_mask].index
    if len(sell_entries_idx) > 0:
        sell_entry_prices = sig.loc[sell_entries_idx, "price"]
        ax.scatter(sell_entries_idx, sell_entry_prices, marker="v", color="red", 
                  s=120, zorder=15, label=f"Sell Entry ({len(sell_entries_idx)})", 
                  edgecolors="darkred", linewidths=2, alpha=0.9)
    
    # Long exits: position was 1, now is 0 or -1
    long_exits_mask = (prev_position == 1) & (sig["position"] != 1)
    long_exits_idx = long_exits_mask[long_exits_mask].index
    if len(long_exits_idx) > 0:
        long_exit_prices = sig.loc[long_exits_idx, "price"]
        ax.scatter(long_exits_idx, long_exit_prices, marker="X", color="orange", 
                  s=100, zorder=15, label=f"Long Exit ({len(long_exits_idx)})", 
                  edgecolors="darkorange", linewidths=1.5, alpha=0.85)
    
    # Short exits: position was -1, now is 0 or 1
    short_exits_mask = (prev_position == -1) & (sig["position"] != -1)
    short_exits_idx = short_exits_mask[short_exits_mask].index
    if len(short_exits_idx) > 0:
        short_exit_prices = sig.loc[short_exits_idx, "price"]
        ax.scatter(short_exits_idx, short_exit_prices, marker="X", color="orange", 
                  s=100, zorder=15, label=f"Short Exit ({len(short_exits_idx)})", 
                  edgecolors="darkorange", linewidths=1.5, alpha=0.85)
    
    ax.set_title(f"{ticker} — Price & EMAs with Trading Signals\n"
                f"Green Background=Long | Red Background=Short | "
                f"↑Buy Entry (lime) | ↓Sell Entry (red) | ✗ Exit (orange)", 
                fontsize=11, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()

    # 2️⃣ VOLATILITIES
    plt.figure(figsize=(10, 4))
    plt.plot(sig.index, sig["garch_vol"], label="GARCH Vol")
    plt.plot(sig.index, sig["heston_vol"], label="Heston Vol")
    plt.plot(sig.index, sig["vol_pred_cons"], label="Consensus Vol")
    vol_benchmark_col = "vol_benchmark" if "vol_benchmark" in sig.columns else [c for c in sig.columns if c.startswith("vol_hist_")][0]
    plt.plot(sig.index, sig[vol_benchmark_col], label="Benchmark Vol (MA7 de Vol21d)")
    if "vol_forecast_final" in sig.columns:
        plt.plot(sig.index, sig["vol_forecast_final"], label="Vol Forecast Final (Consenso Inteligente)", linestyle="--", alpha=0.8)
    plt.title(f"{ticker} — Predicted vs Realized Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3️⃣ Z-SCORE (Meta-Estratégia Assimétrica 2.0)
    plt.figure(figsize=(12, 5))
    plt.plot(sig.index, sig["zscore"], label="Z-Score", color="blue", lw=1.2)
    
    # Show buy and sell thresholds (Assimétricos)
    buy_thresh = -0.5  # COMPRA: Z < -0.5 (Calmaria)
    sell_thresh = 1.5  # VENDA: Z > 1.5 (Medo Real)
    override_thresh = 1.0  # OVERRIDE: até Z < 1.0 com divergência > 1%
    exit_short_thresh = 0.5  # Short Exit: Z < 0.5
    
    plt.axhline(buy_thresh, color="green", linestyle="--", linewidth=1.5, 
                label=f"Buy Threshold ({buy_thresh:.1f}) - Calmaria")
    plt.axhline(sell_thresh, color="red", linestyle="--", linewidth=1.5, 
                label=f"Sell Threshold ({sell_thresh:.1f}) - Medo Real")
    plt.axhline(override_thresh, color="orange", linestyle=":", linewidth=1, alpha=0.7,
                label=f"Override Max ({override_thresh:.1f}) - Divergência > 1%")
    plt.axhline(exit_short_thresh, color="purple", linestyle=":", linewidth=1, alpha=0.7,
                label=f"Short Exit ({exit_short_thresh:.1f}) - Saída de Pânico")
    plt.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    
    # Shade regions
    plt.fill_between(sig.index, sig["zscore"].min(), buy_thresh, 
                     where=(sig["zscore"] < buy_thresh), alpha=0.1, color="green", label="Calmaria (Buy Zone)")
    plt.fill_between(sig.index, sell_thresh, sig["zscore"].max(), 
                     where=(sig["zscore"] > sell_thresh), alpha=0.1, color="red", label="Medo Real (Sell Zone)")
    
    plt.title(f"{ticker} — Z-Score (Meta-Estratégia Assimétrica 2.0)")
    plt.ylabel("Z-Score")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
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


def visualize_ema_comparison(sig_meta: pd.DataFrame, sig_ema: pd.DataFrame, cfg, ticker: str) -> None:
    """
    Compara a meta-estratégia (com modelo de volatilidade) vs estratégia apenas com EMAs.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Calcular retornos e equity para ambas estratégias
    strat_ret_meta = _strategy_returns(sig_meta)
    equity_meta = _equity_curve(strat_ret_meta)
    dd_meta = _drawdown(equity_meta)
    
    strat_ret_ema = _strategy_returns(sig_ema)
    equity_ema = _equity_curve(strat_ret_ema)
    dd_ema = _drawdown(equity_ema)
    
    # Alinhar índices
    idx_common = equity_meta.index.intersection(equity_ema.index)
    equity_meta = equity_meta.reindex(idx_common)
    equity_ema = equity_ema.reindex(idx_common)
    dd_meta = dd_meta.reindex(idx_common)
    dd_ema = dd_ema.reindex(idx_common)
    
    # Métricas para comparação
    total_return_meta = equity_meta.iloc[-1] - 1.0
    total_return_ema = equity_ema.iloc[-1] - 1.0
    cagr_meta = (equity_meta.iloc[-1] ** (252 / max(1, len(idx_common)))) - 1.0
    cagr_ema = (equity_ema.iloc[-1] ** (252 / max(1, len(idx_common)))) - 1.0
    sharpe_meta = (strat_ret_meta.reindex(idx_common).mean() / strat_ret_meta.reindex(idx_common).std()) * np.sqrt(252) if strat_ret_meta.reindex(idx_common).std() > 0 else np.nan
    sharpe_ema = (strat_ret_ema.reindex(idx_common).mean() / strat_ret_ema.reindex(idx_common).std()) * np.sqrt(252) if strat_ret_ema.reindex(idx_common).std() > 0 else np.nan
    hit_meta = _hit_rate(strat_ret_meta.reindex(idx_common), sig_meta["returns"].reindex(idx_common))
    hit_ema = _hit_rate(strat_ret_ema.reindex(idx_common), sig_ema["returns"].reindex(idx_common))
    
    # Contar trades
    buy_entries_meta = int((sig_meta['position'].reindex(idx_common).diff() == 1).sum())
    sell_entries_meta = int((sig_meta['position'].reindex(idx_common).diff() == -1).sum())
    total_trades_meta = buy_entries_meta + sell_entries_meta
    
    buy_entries_ema = int((sig_ema['position'].reindex(idx_common).diff() == 1).sum())
    sell_entries_ema = int((sig_ema['position'].reindex(idx_common).diff() == -1).sum())
    total_trades_ema = buy_entries_ema + sell_entries_ema
    
    # Print comparação
    print("\n" + "=" * 80)
    print("COMPARAÇÃO: META-ESTRATÉGIA vs ESTRATÉGIA APENAS COM EMAs")
    print("=" * 80)
    print(f"{'Métrica':<30} {'Meta-Estratégia':<20} {'EMA-Only':<20} {'Diferença':<15}")
    print("-" * 80)
    print(f"{'Total Return':<30} {total_return_meta:>18.2%} {total_return_ema:>18.2%} {(total_return_meta - total_return_ema):>13.2%}")
    print(f"{'CAGR (approx)':<30} {cagr_meta:>18.2%} {cagr_ema:>18.2%} {(cagr_meta - cagr_ema):>13.2%}")
    print(f"{'Sharpe Ratio':<30} {sharpe_meta:>18.2f} {sharpe_ema:>18.2f} {(sharpe_meta - sharpe_ema):>13.2f}")
    print(f"{'Hit Rate':<30} {hit_meta:>18.2%} {hit_ema:>18.2%} {(hit_meta - hit_ema):>13.2%}")
    print(f"{'Total Trades':<30} {total_trades_meta:>18d} {total_trades_ema:>18d} {(total_trades_meta - total_trades_ema):>13d}")
    print(f"{'Buy Entries':<30} {buy_entries_meta:>18d} {buy_entries_ema:>18d} {(buy_entries_meta - buy_entries_ema):>13d}")
    print(f"{'Sell Entries':<30} {sell_entries_meta:>18d} {sell_entries_ema:>18d} {(sell_entries_meta - sell_entries_ema):>13d}")
    print("=" * 80)
    
    # Gráfico comparativo
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 1) Price + EMAs com posições
    ax1 = axes[0]
    ax1.plot(sig_meta.index, sig_meta["price"], label="Price", lw=1.5, color="black", zorder=5)
    ax1.plot(sig_meta.index, sig_meta[f"ema{cfg.ema_fast}"], label=f"EMA {cfg.ema_fast}", lw=1.2, alpha=0.8)
    ax1.plot(sig_meta.index, sig_meta[f"ema{cfg.ema_slow}"], label=f"EMA {cfg.ema_slow}", lw=1.2, alpha=0.8)
    
    # Shade positions for meta-strategy
    long_mask_meta = sig_meta["position"] == 1
    short_mask_meta = sig_meta["position"] == -1
    ax1.fill_between(sig_meta.index, sig_meta["price"].min(), sig_meta["price"].max(),
                     where=long_mask_meta, color="green", alpha=0.15, label="Meta: Long", zorder=1)
    ax1.fill_between(sig_meta.index, sig_meta["price"].min(), sig_meta["price"].max(),
                     where=short_mask_meta, color="red", alpha=0.15, label="Meta: Short", zorder=1)
    
    # Shade positions for EMA-only strategy (lighter)
    long_mask_ema = sig_ema["position"] == 1
    short_mask_ema = sig_ema["position"] == -1
    ax1.fill_between(sig_ema.index, sig_ema["price"].min(), sig_ema["price"].max(),
                     where=long_mask_ema, color="cyan", alpha=0.08, label="EMA-Only: Long", zorder=0)
    ax1.fill_between(sig_ema.index, sig_ema["price"].min(), sig_ema["price"].max(),
                     where=short_mask_ema, color="magenta", alpha=0.08, label="EMA-Only: Short", zorder=0)
    
    ax1.set_title(f"{ticker} — Price & EMAs: Meta-Estratégia (verde/vermelho) vs EMA-Only (ciano/magenta)", 
                  fontsize=11, pad=10)
    ax1.set_ylabel("Price")
    ax1.legend(loc="best", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle="--")
    
    # 2) Equity curves comparison
    ax2 = axes[1]
    ax2.plot(idx_common, equity_meta, label=f"Meta-Estratégia (Return: {total_return_meta:.2%})", 
             lw=2, color="blue", alpha=0.8)
    ax2.plot(idx_common, equity_ema, label=f"EMA-Only (Return: {total_return_ema:.2%})", 
             lw=2, color="orange", alpha=0.8)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.set_title("Equity Curves Comparison", fontsize=11, pad=10)
    ax2.set_ylabel("Cumulative Return")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--")
    
    # 3) Drawdown comparison
    ax3 = axes[2]
    ax3.fill_between(idx_common, dd_meta, 0, alpha=0.5, color="blue", label="Meta-Estratégia Drawdown")
    ax3.fill_between(idx_common, dd_ema, 0, alpha=0.5, color="orange", label="EMA-Only Drawdown")
    ax3.set_title("Drawdown Comparison", fontsize=11, pad=10)
    ax3.set_ylabel("Drawdown")
    ax3.set_xlabel("Date")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.show()

def run() -> None:
    print("=" * 60)
    print("META-ESTRATÉGIA DE TRADING - ANÁLISE DE VOLATILIDADE")
    print("=" * 60)
    print(f"Ativo: {TICKER}")
    print(f"Período: {START} até hoje")
    print(f"Perfil: {PROFILE}")
    print("=" * 60)
    
    # 1) Load prices from Yahoo Finance
    print(f"\n[1/5] Carregando dados de preços para {TICKER}...")
    prices = load_prices_yf(TICKER, start=START)
    print(f"  ✓ {len(prices)} dias de dados carregados")

    # 2) Returns
    print(f"\n[2/5] Calculando retornos diários...")
    returns = prices.pct_change().dropna()
    print(f"  ✓ {len(returns)} retornos calculados")

    # 3) Volatility estimates (daily, same index/scale)
    print(f"\n[3/5] Estimando volatilidade com modelos Heston e GARCH...")
    print("  → Heston (Monte Carlo)...")
    heston_vol = estimate_heston_vol(returns, ticker=TICKER, verbose=True)
    print(f"  ✓ Heston: {len(heston_vol.dropna())} previsões válidas")
    
    print("  → GARCH...")
    garch_vol = estimate_garch_vol(returns, variant="GARCH")
    print(f"  ✓ GARCH: {len(garch_vol.dropna())} previsões válidas")

    # Align indices
    print(f"\n[4/5] Alinhando índices e calculando consenso...")
    idx = prices.index.intersection(heston_vol.index).intersection(garch_vol.index)
    prices     = prices.reindex(idx)
    heston_vol = heston_vol.reindex(idx)
    garch_vol  = garch_vol.reindex(idx)
    
    # Show consensus calculation
    consensus_vol = (heston_vol + garch_vol) / 2.0
    print(f"  ✓ Consenso de Volatilidade = (Heston + GARCH) / 2")
    print(f"    Média Heston: {heston_vol.mean():.6f}")
    print(f"    Média GARCH:  {garch_vol.mean():.6f}")
    print(f"    Média Consenso: {consensus_vol.mean():.6f}")

    # 4) Build signals (Slides 4→9)
    print(f"\n[5/5] Construindo sinais de trading...")
    cfg = SignalConfig(profile=PROFILE)
    sig = build_signals(prices, garch_vol, heston_vol, cfg=cfg)
    print(f"  ✓ {len(sig)} sinais gerados")

    # 5) Save output
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True, parents=True)
    out_csv = outdir / "signals.csv"
    sig.to_csv(out_csv, index=True)

    # 6) Quick summary
    print("\n" + "=" * 60)
    print("RESUMO DOS SINAIS")
    print("=" * 60)
    print(f"Ativo: {TICKER}")
    print(f"Período: {START} até {sig.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total de dias: {len(sig)}")
    z_thresh = cfg.z_thresholds[PROFILE]
    print(f"Perfil: {PROFILE} (Meta-Estratégia Assimétrica 2.0)")
    print(f"  Buy Threshold: Z < -0.5 (Calmaria) + OVERRIDE se divergência > 1%")
    print(f"  Sell Threshold: Z > 1.5 (Medo Real)")
    print(f"  Long Exit: EMA 7 cruza abaixo EMA 21")
    print(f"  Short Exit: EMA 7 cruza acima EMA 21 OU Z < 0.5")
    print(f"\nEstatísticas de Volatilidade:")
    print(f"  Heston (média): {sig['heston_vol'].mean():.6f}")
    print(f"  GARCH (média):  {sig['garch_vol'].mean():.6f}")
    print(f"  Consenso (média): {sig['vol_pred_cons'].mean():.6f}")
    vol_benchmark_col = "vol_benchmark" if "vol_benchmark" in sig.columns else [c for c in sig.columns if c.startswith("vol_hist_")][0]
    print(f"  Benchmark (média): {sig[vol_benchmark_col].mean():.6f}")
    if "vol_forecast_final" in sig.columns:
        print(f"  Forecast Final (média): {sig['vol_forecast_final'].mean():.6f}")
    print(f"\nZ-Score:")
    print(f"  Média: {sig['zscore'].mean():.2f}")
    print(f"  Std:   {sig['zscore'].std():.2f}")
    print(f"  Min:   {sig['zscore'].min():.2f}")
    print(f"  Max:   {sig['zscore'].max():.2f}")
    print(f"\nPosições:")
    buy_entries = int((sig['position'].diff() == 1).sum())
    sell_entries = int((sig['position'].diff() == -1).sum())
    days_long = int((sig['position'] == 1).sum())
    days_short = int((sig['position'] == -1).sum())
    print(f"  Entradas de COMPRA: {buy_entries}")
    print(f"  Entradas de VENDA: {sell_entries}")
    print(f"  Dias em LONG: {days_long} ({days_long/len(sig)*100:.1f}%)")
    print(f"  Dias em SHORT: {days_short} ({days_short/len(sig)*100:.1f}%)")
    print(f"  Dias NEUTRO: {len(sig) - days_long - days_short} ({(len(sig) - days_long - days_short)/len(sig)*100:.1f}%)")
    print(f"\nArquivo salvo: {out_csv}")
    print("=" * 60)
    
    print("\nÚltimas 10 linhas:")
    cols = ["price", "garch_vol", "heston_vol", "vol_pred_cons"]
    if "vol_forecast_final" in sig.columns:
        cols.append("vol_forecast_final")
    cols.extend([vol_benchmark_col, "zscore", "risk_state", 
            "ema_cross_up", "ema_cross_down",
            "buy_gate", "sell_gate", "buy_signal", "sell_signal", "position"])
    available_cols = [c for c in cols if c in sig.columns]
    print(sig[available_cols].tail(10).to_string())

    # 7) Build EMA-only strategy for comparison
    print("\n" + "=" * 60)
    print("Construindo estratégia apenas com EMAs (para comparação)...")
    print("=" * 60)
    sig_ema_only = build_ema_only_signals(prices, ema_fast=cfg.ema_fast, ema_slow=cfg.ema_slow)
    print(f"  ✓ {len(sig_ema_only)} sinais gerados (EMA-only)")

    # 8) Visualization & metrics
    print("\n" + "=" * 60)
    print("Gerando visualizações e métricas...")
    print("=" * 60)
    visualize(sig, cfg, TICKER)
    
    # 9) Comparison visualization
    print("\n" + "=" * 60)
    print("Gerando comparação: Meta-Estratégia vs EMA-Only...")
    print("=" * 60)
    visualize_ema_comparison(sig, sig_ema_only, cfg, TICKER)
    
if __name__ == "__main__":
    run()

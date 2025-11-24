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


def build_signals_ema_only(sig: pd.DataFrame, cfg) -> pd.Series:
    """
    Gera sinais de trading apenas com cruzamento de médias (EMA rápida vs EMA lenta).
    - COMPRA: Quando EMA rápida cruza acima da EMA lenta (golden cross)
    - VENDA: Quando EMA rápida cruza abaixo da EMA lenta (death cross)
    - Mantém posição enquanto a tendência continua
    """
    ema_fast_col = f"ema{cfg.ema_fast}"
    ema_slow_col = f"ema{cfg.ema_slow}"
    
    # Detectar cruzamentos
    # Golden cross: EMA rápida cruza acima da EMA lenta
    golden_cross = (sig[ema_fast_col] > sig[ema_slow_col]) & (sig[ema_fast_col].shift(1) <= sig[ema_slow_col].shift(1))
    # Death cross: EMA rápida cruza abaixo da EMA lenta
    death_cross = (sig[ema_fast_col] < sig[ema_slow_col]) & (sig[ema_fast_col].shift(1) >= sig[ema_slow_col].shift(1))
    
    # Tendência atual
    trend_up = sig[ema_fast_col] > sig[ema_slow_col]
    trend_down = sig[ema_fast_col] < sig[ema_slow_col]
    
    # Inicializar posições
    position_ema = pd.Series(0, index=sig.index, dtype=int, name="position_ema_only")
    
    # Lógica de posicionamento: mantém posição enquanto a tendência continua
    for i in range(1, len(sig.index)):
        prev_pos = position_ema.iat[i - 1]
        
        if prev_pos == 1:  # Em posição de COMPRA
            # Mantém compra enquanto trend_up, fecha se trend_down
            position_ema.iat[i] = 1 if trend_up.iat[i] else 0
        elif prev_pos == -1:  # Em posição de VENDA
            # Mantém venda enquanto trend_down, fecha se trend_up
            position_ema.iat[i] = -1 if trend_down.iat[i] else 0
        else:  # Neutro (0)
            # Entra em compra ou venda baseado nos cruzamentos
            if golden_cross.iat[i]:
                position_ema.iat[i] = 1
            elif death_cross.iat[i]:
                position_ema.iat[i] = -1
            else:
                position_ema.iat[i] = 0
    
    # Warm-up: desabilitar posições iniciais onde não há dados suficientes
    warmup_mask = sig[ema_fast_col].isna() | sig[ema_slow_col].isna()
    position_ema.loc[warmup_mask] = 0
    
    return position_ema


def visualize(sig: pd.DataFrame, cfg, ticker: str) -> None:
    """
    Save charts to 'outputs/' and print performance metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True, parents=True)

    # --- metrics prep (estratégia atual com volatilidade) ---
    strat_ret = _strategy_returns(sig)
    equity = _equity_curve(strat_ret)
    dd = _drawdown(equity)
    
    # --- estratégia apenas com médias ---
    position_ema_only = build_signals_ema_only(sig, cfg)
    asset_ret = sig["returns"].fillna(0.0)
    pos_ema = position_ema_only.shift(1).fillna(0).astype(float)
    strat_ret_ema = (asset_ret * pos_ema).rename("strategy_ret_ema_only")
    equity_ema = _equity_curve(strat_ret_ema)
    dd_ema = _drawdown(equity_ema)

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
    
    # 1️⃣ PRICE + EMAs + POSITION (com marcadores de entrada/saída)
    # Criar 2 subplots: Preço (Principal) e Regime de Risco (Secundário)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # --- PLOT 1: PREÇO ---
    ax1.plot(sig.index, sig["price"], label="Price", lw=1.5, color="black", zorder=5)
    ax1.plot(sig.index, sig[f"ema{cfg.ema_fast}"], label=f"EMA {cfg.ema_fast}", lw=1.2, alpha=0.8)
    ax1.plot(sig.index, sig[f"ema{cfg.ema_slow}"], label=f"EMA {cfg.ema_slow}", lw=1.2, alpha=0.8)
    
    # Shade long positions (green) and short positions (red)
    long_mask = sig["position"] == 1
    short_mask = sig["position"] == -1
    ax1.fill_between(sig.index, sig["price"].min(), sig["price"].max(),
                    where=long_mask, color="green", alpha=0.12, label="Long Position", zorder=1)
    
    # Detect entry and exit points
    position_diff = sig["position"].diff()
    prev_position = sig["position"].shift(1).fillna(0).astype(int)
    
    # Buy entries
    buy_entries_mask = (position_diff == 1) & (prev_position != 1)
    buy_entries_idx = buy_entries_mask[buy_entries_mask].index
    if len(buy_entries_idx) > 0:
        buy_entry_prices = sig.loc[buy_entries_idx, "price"]
        ax1.scatter(buy_entries_idx, buy_entry_prices, marker="^", color="lime", 
                  s=120, zorder=15, label=f"Buy Entry ({len(buy_entries_idx)})", 
                  edgecolors="darkgreen", linewidths=2, alpha=0.9)
    
    # Exits (Long)
    long_exits_mask = (prev_position == 1) & (sig["position"] != 1)
    long_exits_idx = long_exits_mask[long_exits_mask].index
    if len(long_exits_idx) > 0:
        long_exit_prices = sig.loc[long_exits_idx, "price"]
        ax1.scatter(long_exits_idx, long_exit_prices, marker="X", color="orange", 
                  s=100, zorder=15, label=f"Exit ({len(long_exits_idx)})", 
                  edgecolors="darkorange", linewidths=1.5, alpha=0.85)
    
    ax1.set_title(f"{ticker} — Price & EMAs with Trading Signals\nDynamic Exit Strategy: Calm=Slow Exit | Risky=Fast Exit", 
                fontsize=11, pad=10)
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # --- PLOT 2: REGIME DE RISCO (Z-Score Downside) ---
    z_down = sig["zscore_downside"]
    ax2.plot(sig.index, z_down, label="Downside Z-Score", color="purple", lw=1.5)
    ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
    
    # Colorir fundo baseado no regime
    # Risco (Z > 0): Fundo Vermelho Claro
    # Calmaria (Z <= 0): Fundo Azul Claro
    ax2.fill_between(sig.index, z_down.min(), z_down.max(), where=(z_down > 0), 
                     color="red", alpha=0.1, label="Risky Regime (Fast Exit)")
    ax2.fill_between(sig.index, z_down.min(), z_down.max(), where=(z_down <= 0), 
                     color="blue", alpha=0.1, label="Calm Regime (Slow Exit)")
    
    ax2.set_title("Risk Regime (Downside Volatility)", fontsize=10)
    ax2.set_ylabel("Z-Score")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(outdir / "chart_price_signals.png")
    plt.show()

    # 2️⃣ VOLATILITIES
    plt.figure(figsize=(10, 4))
    plt.plot(sig.index, sig["garch_vol"], label="GARCH Vol")
    plt.plot(sig.index, sig["heston_vol"], label="Heston Vol")
    plt.plot(sig.index, sig["vol_pred_cons"], label="Consensus Vol")
    vol_hist_col = [c for c in sig.columns if c.startswith("vol_hist_")][0]
    plt.plot(sig.index, sig[vol_hist_col], label="Benchmark Vol (MA7 de Vol21d)")
    plt.title(f"{ticker} — Predicted vs Realized Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "chart_volatility.png")
    plt.show()

    # 3️⃣ Z-SCORE
    plt.figure(figsize=(12, 5))
    plt.plot(sig.index, sig["zscore"], label="Z-Score", color="blue", lw=1.2)
    
    # Show buy and sell thresholds
    z_thresh = cfg.z_thresholds[cfg.profile]
    buy_thresh = z_thresh["buy"]
    sell_thresh = z_thresh["sell"]
    
    plt.axhline(buy_thresh, color="green", linestyle="--", linewidth=1.5, 
                label=f"Buy Threshold ({buy_thresh:.1f})")
    plt.axhline(sell_thresh, color="red", linestyle="--", linewidth=1.5, 
                label=f"Sell Threshold ({sell_thresh:.1f})")
    plt.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    
    # Shade regions
    plt.fill_between(sig.index, sig["zscore"].min(), buy_thresh, 
                     where=(sig["zscore"] < buy_thresh), alpha=0.1, color="green", label="Calmaria (Buy Zone)")
    plt.fill_between(sig.index, sell_thresh, sig["zscore"].max(), 
                     where=(sig["zscore"] > sell_thresh), alpha=0.1, color="red", label="Risco (Sell Zone)")
    
    plt.title(f"{ticker} — Z-Score ({cfg.profile} profile)")
    plt.ylabel("Z-Score")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "chart_zscore.png")
    plt.show()

    # 4️⃣ EQUITY & DRAWDOWN
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(sig.index, equity, color="blue")
    ax[0].set_title("Strategy Equity")
    ax[1].plot(sig.index, dd, color="red")
    ax[1].set_title("Drawdown")
    plt.tight_layout()
    plt.savefig(outdir / "chart_equity_drawdown.png")
    plt.show()

    # 5️⃣ HISTOGRAM
    plt.figure(figsize=(7, 4))
    plt.hist(strat_ret.dropna(), bins=40, alpha=0.7)
    plt.title(f"{ticker} — Strategy Daily Returns Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "chart_returns_dist.png")
    plt.show()

    # 6️⃣ COMPARAÇÃO: RETORNO ACUMULADO - Estratégia com Volatilidade vs Apenas Médias
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot equity curves
    ax.plot(sig.index, equity, label="Estratégia com Volatilidade (Heston+GARCH)", 
            color="blue", lw=2, alpha=0.9)
    ax.plot(sig.index, equity_ema, label="Estratégia Apenas Médias (EMA Cross)", 
            color="orange", lw=2, alpha=0.9)
    
    # Linha de referência (buy and hold = 1.0)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Buy & Hold (1.0)")
    
    # Calcular métricas para comparação
    total_return_vol = equity.iloc[-1] - 1.0
    total_return_ema = equity_ema.iloc[-1] - 1.0
    cagr_vol = (equity.iloc[-1] ** (252 / max(1, len(sig)))) - 1.0
    cagr_ema = (equity_ema.iloc[-1] ** (252 / max(1, len(sig)))) - 1.0
    sharpe_vol = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() > 0 else np.nan
    sharpe_ema = (strat_ret_ema.mean() / strat_ret_ema.std()) * np.sqrt(252) if strat_ret_ema.std() > 0 else np.nan
    
    # Adicionar texto com métricas
    metrics_text = (
        f"Estratégia com Volatilidade:\n"
        f"  Retorno Total: {total_return_vol:.2%}\n"
        f"  CAGR: {cagr_vol:.2%}\n"
        f"  Sharpe: {sharpe_vol:.2f}\n\n"
        f"Estratégia Apenas Médias:\n"
        f"  Retorno Total: {total_return_ema:.2%}\n"
        f"  CAGR: {cagr_ema:.2%}\n"
        f"  Sharpe: {sharpe_ema:.2f}\n\n"
        f"Diferença: {total_return_vol - total_return_ema:+.2%}"
    )
    
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(f"{ticker} — Comparação de Retorno Acumulado\n"
                f"Estratégia com Previsão de Volatilidade vs Estratégia Apenas com Médias",
                fontsize=12, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (Normalizado em 1.0)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(outdir / "chart_comparison.png")
    plt.show()
    
    # Métricas comparativas no console
    print("\n" + "=" * 60)
    print("COMPARAÇÃO DE ESTRATÉGIAS")
    print("=" * 60)
    print(f"\n📊 ESTRATÉGIA COM VOLATILIDADE (Heston + GARCH):")
    print(f"   Retorno Total       : {total_return_vol: .2%}")
    print(f"   CAGR (approx)       : {cagr_vol: .2%}")
    print(f"   Sharpe (no RF)      : {sharpe_vol: .2f}")
    print(f"   Volatilidade (ann)  : {strat_ret.std() * np.sqrt(252): .2%}")
    print(f"   Hit Rate            : {_hit_rate(strat_ret, sig['returns']): .2%}")
    
    # Contar trades da estratégia apenas com médias
    buy_entries_ema = int((position_ema_only.diff() == 1).sum())
    sell_entries_ema = int((position_ema_only.diff() == -1).sum())
    days_long_ema = int((position_ema_only == 1).sum())
    days_short_ema = int((position_ema_only == -1).sum())
    
    print(f"\n📈 ESTRATÉGIA APENAS COM MÉDIAS (EMA Cross):")
    print(f"   Retorno Total       : {total_return_ema: .2%}")
    print(f"   CAGR (approx)       : {cagr_ema: .2%}")
    print(f"   Sharpe (no RF)      : {sharpe_ema: .2f}")
    print(f"   Volatilidade (ann)  : {strat_ret_ema.std() * np.sqrt(252): .2%}")
    print(f"   Hit Rate            : {_hit_rate(strat_ret_ema, sig['returns']): .2%}")
    print(f"   Total Trades        : {buy_entries_ema + sell_entries_ema} (Compras: {buy_entries_ema}, Vendas: {sell_entries_ema})")
    print(f"   Days in Position    : Long={days_long_ema}, Short={days_short_ema}, Neutral={len(sig) - days_long_ema - days_short_ema}")
    
    print(f"\n🔍 DIFERENÇA:")
    print(f"   Retorno             : {total_return_vol - total_return_ema:+.2%} "
          f"({'Melhor com Volatilidade' if total_return_vol > total_return_ema else 'Melhor com Médias'})")
    print(f"   Sharpe              : {sharpe_vol - sharpe_ema:+.2f} "
          f"({'Melhor com Volatilidade' if sharpe_vol > sharpe_ema else 'Melhor com Médias'})")
    print("=" * 60)

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
    print(f"Perfil: {PROFILE} (Buy: {z_thresh['buy']:.1f}, Sell: {z_thresh['sell']:.1f})")
    print(f"\nEstatísticas de Volatilidade:")
    print(f"  Heston (média): {sig['heston_vol'].mean():.6f}")
    print(f"  GARCH (média):  {sig['garch_vol'].mean():.6f}")
    print(f"  Consenso (média): {sig['vol_pred_cons'].mean():.6f}")
    vol_hist_col = [c for c in sig.columns if c.startswith("vol_hist_")][0]
    print(f"  Histórica (média): {sig[vol_hist_col].mean():.6f}")
    print(f"  Downside (média):  {sig['vol_downside'].mean():.6f}")
    print(f"\nZ-Score:")
    print(f"  Total (Média):    {sig['zscore'].mean():.2f}")
    print(f"  Downside (Média): {sig['zscore_downside'].mean():.2f}")
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
    cols = ["price", "garch_vol", "heston_vol", "vol_pred_cons", vol_hist_col, "zscore", "risk_state", 
            "ema_fast_slope", "ema_fast_strong_up", "ema_fast_strong_down",
            "buy_gate", "sell_gate", "buy_signal", "sell_signal", "position"]
    print(sig[cols].tail(10).to_string())

    # 7) Visualization & metrics
    print("\n" + "=" * 60)
    print("Gerando visualizações e métricas...")
    print("=" * 60)
    visualize(sig, cfg, TICKER)
    
if __name__ == "__main__":
    run()

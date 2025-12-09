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
        df.columns = df.columns.get_level_values(0)
    
    df.index.name = "Date"

    # prefer Adj Close; fallback to Close
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].copy()
    s.name = "Adj Close"
    
    # Ensure index is a simple DatetimeIndex
    if isinstance(s.index, pd.MultiIndex):
        s.index = s.index.get_level_values(0)
    
    return s

def _strategy_returns(sig: pd.DataFrame) -> pd.Series:
    """Daily strategy returns using long/short positions with 1-day lag on execution."""
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

def plot_interactive_trades(sig: pd.DataFrame, price: pd.Series, title: str):
    """
    Plota preço e sinais de trade de forma interativa (com zoom).
    Verde = Compra, Vermelho = Venda.
    """
    plt.figure(figsize=(14, 7))
    
    # Preço base
    plt.plot(price.index, price, label="Preço", color="black", alpha=0.6, linewidth=1)
    
    # EMAs (para referência visual de tendência)
    ema_fast = price.ewm(span=7, adjust=False).mean()
    ema_slow = price.ewm(span=21, adjust=False).mean()
    plt.plot(price.index, ema_fast, label="EMA 7", color="orange", alpha=0.4, linewidth=0.8)
    plt.plot(price.index, ema_slow, label="EMA 21", color="blue", alpha=0.4, linewidth=0.8)

    # Identificar mudanças de posição (Trades)
    pos = sig["position"]
    trades = pos.diff().fillna(0)
    
    # Compras (Entradas Long ou Fechamento de Short para Long)
    # Valor +1 ou +2 (se reverteu de -1 para 1)
    # Mas aqui queremos saber quando ESTAMOS comprados.
    
    # Vamos marcar os PONTOS onde a posição MUDA.
    # Compra Nova: (antigo <= 0) e (novo == 1)
    buy_entries = (pos == 1) & (pos.shift(1) <= 0)
    # Venda Nova (Short): (antigo >= 0) e (novo == -1)
    short_entries = (pos == -1) & (pos.shift(1) >= 0)
    # Saídas (Flat): (novo == 0) e (antigo != 0)
    # exits = (pos == 0) & (pos.shift(1) != 0)
    
    # Marcar no gráfico
    # Usar o preço do dia do sinal (na prática executa no próximo, mas marcador visual fica no dia)
    plt.scatter(price.index[buy_entries], price[buy_entries], 
                marker="^", color="green", s=100, label="Compra (Long)", zorder=10)
    
    plt.scatter(price.index[short_entries], price[short_entries], 
                marker="v", color="red", s=100, label="Venda (Short)", zorder=10)

    # Preencher fundo com a cor da posição ATUAL
    # Verde fraco para Long, Vermelho fraco para Short
    # Criar coleção de spans é pesado, então vamos pintar por 'fill_between' condicional
    # Truque: pintar área inteira onde pos == 1
    y_min, y_max = price.min(), price.max()
    plt.fill_between(price.index, y_min, y_max, where=(pos == 1), 
                     color='green', alpha=0.05, label="Em Posição Long")
    plt.fill_between(price.index, y_min, y_max, where=(pos == -1), 
                     color='red', alpha=0.05, label="Em Posição Short")

    plt.title(title, fontsize=14)
    plt.ylabel("Preço ($)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def run_strategy_suite():
    print("=" * 80)
    print("SISTEMA CENTRALIZADO: META-ESTRATÉGIA (MAIN.PY)")
    print("=" * 80)

    # 1. Dados e Modelagem
    print(f"\n[1/4] Carregando dados e calculando volatilidades para {TICKER}...")
    prices = load_prices_yf(TICKER, start=START)
    returns = prices.pct_change().dropna()
    
    # Volatilidade Passada (L2) - Suavizada e Shiftada
    vol_realized_21d = returns.rolling(window=21).std() * np.sqrt(252)
    vol_input_lvl2 = (vol_realized_21d.shift(1).fillna(method='bfill')) / np.sqrt(252)
    
    # Volatilidade Futura (L4) - Oráculo
    vol_future_pure = returns.rolling(window=5).std() * np.sqrt(252)
    vol_input_lvl4 = (vol_future_pure.shift(-5).fillna(method='ffill')) / np.sqrt(252)
    
    # Volatilidades dos Modelos (L3)
    print("  Estimando Heston e GARCH (pode demorar)...")
    heston_vol = estimate_heston_vol(returns, ticker=TICKER, verbose=False)
    garch_vol = estimate_garch_vol(returns, variant="GARCH")
    
    # Alinhamento
    idx = prices.index.intersection(heston_vol.index).intersection(garch_vol.index).intersection(vol_realized_21d.index)
    prices = prices.reindex(idx)
    returns = returns.reindex(idx)
    heston_vol = heston_vol.reindex(idx)
    garch_vol = garch_vol.reindex(idx)
    vol_input_lvl2 = vol_input_lvl2.reindex(idx)
    vol_input_lvl4 = vol_input_lvl4.reindex(idx)

    # 2. Execução (4 Níveis)
    print("\n[2/4] Executando Estratégias (L1 a L4)...")
    
    # Configs
    cfg_base = SignalConfig(profile="moderate") # Padrão
    cfg_aggressive = SignalConfig(profile="aggressive", panic_z_factor=2.5) # Otimizado L3
    
    # Oráculo Standard: Sem Panic Override (apenas filtro de tendência)
    # Definimos panic_factor altíssimo para garantir que ele nunca venda "em pânico",
    # mas apenas quando a tendência de preço reverter.
    cfg_oracle_standard = SignalConfig(profile="moderate", panic_z_factor=100.0)

    # L1: EMA Only
    print("  Processing Level 1 (EMA Only)...")
    sig_lvl1 = build_ema_only_signals(prices, ema_fast=7, ema_slow=21)
    
    # L2: Vol Passada
    print("  Processing Level 2 (Past Vol)...")
    sig_lvl2 = build_signals(prices, garch_vol=vol_input_lvl2, heston_vol=vol_input_lvl2, cfg=cfg_base)
    
    # L3: Modelos (Otimizada v6)
    print("  Processing Level 3 (Models - Aggressive)...")
    sig_lvl3 = build_signals(prices, garch_vol=garch_vol, heston_vol=heston_vol, cfg=cfg_aggressive)
    
    # L4: Oráculo (Standard)
    print("  Processing Level 4 (Oracle Standard)...")
    sig_lvl4 = build_signals(prices, garch_vol=vol_input_lvl4, heston_vol=vol_input_lvl4, cfg=cfg_oracle_standard)

    # 3. Visualização (Plots Interativos)
    print("\n[3/4] Gerando Gráficos Interativos...")
    
    # Plot L1
    plot_interactive_trades(sig_lvl1, prices, "Nível 1: Médias Móveis")
    
    # Plot L2
    plot_interactive_trades(sig_lvl2, prices, "Nível 2:  Vol Passada")
    
    # Plot L3
    plot_interactive_trades(sig_lvl3, prices, "Nível 3: Meta-Estratégia (Heston/GARCH)")
    
    # Plot L4
    plot_interactive_trades(sig_lvl4, prices, "Nível 4: Vol Futura")

    # 4. Comparativo Final (Equity)
    print("\n[4/4] Gerando Comparativo de Equity...")
    plt.figure(figsize=(14, 8))
    
    scenarios = [
        ("Nível 1 (EMA)", sig_lvl1, "gray", ":", 1.5),
        ("Nível 2 (Vol Passada)", sig_lvl2, "skyblue", "-", 1.5),
        ("Nível 3 (Meta-Estratégia)", sig_lvl3, "teal", "-", 2.5),
        ("Nível 4 (Vol Futura)", sig_lvl4, "gold", "--", 1.5)
    ]
    
    print("\nRESULTADOS FINAIS:")
    print(f"{'Estratégia':<25} {'Sharpe':<8} {'Retorno':<10} {'Drawdown':<10}")
    print("-" * 60)
    
    for name, sig, color, style, lw in scenarios:
        strat_ret = _strategy_returns(sig)
        equity = _equity_curve(strat_ret)
        # Normalizar para 1.0
        equity = equity.reindex(idx).fillna(method='ffill').fillna(1.0)
        equity = equity / equity.iloc[0]
        
        total_ret = equity.iloc[-1] - 1.0
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() > 0 else 0
        dd = _drawdown(equity).min()
        
        print(f"{name:<25} {sharpe:<8.2f} {total_ret:<10.2%} {dd:<10.2%}")
        # Converter para Porcentagem
        pct_return = (equity - 1.0) * 100
        
        plt.plot(pct_return.index, pct_return, label=f"{name} (Sharpe: {sharpe:.2f})", color=color, linestyle=style, linewidth=lw)

    plt.title(f"Resultado Final: Retorno Comparativo ({TICKER})", fontsize=16)
    plt.ylabel("Retorno Acumulado (%)")
    plt.xlabel("Ano")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print("\n[PRONTO] Todos os gráficos foram gerados. Eles abrirão em janelas interativas.")
    plt.show()

if __name__ == "__main__":
    try:
        run_strategy_suite()
    except KeyboardInterrupt:
        print("\nUsuário interrompeu a execução.")
    except Exception as e:
        import traceback
        traceback.print_exc()

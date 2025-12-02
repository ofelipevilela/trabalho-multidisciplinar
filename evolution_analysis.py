# evolution_analysis.py

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local modules
from main import load_prices_yf, _strategy_returns, _equity_curve, _drawdown, _rolling_sharpe
from model_heston import estimate_heston_vol
from model_garch import estimate_garch_vol
from signals import SignalConfig, build_signals, build_ema_only_signals

# Config
TICKER = "^GSPC"
START = "2018-01-01"
PROFILE = "moderate"

def run_evolution_analysis():
    print("=" * 80)
    print("ANÁLISE EVOLUTIVA DA ESTRATÉGIA (4 NÍVEIS)")
    print("=" * 80)

    # 1. Preparação dos Dados
    print(f"\n[1/4] Carregando dados e calculando volatilidades para {TICKER}...")
    prices = load_prices_yf(TICKER, start=START)
    returns = prices.pct_change().dropna()
    
    # Volatilidade Realizada 21d (Anualizada)
    vol_realized_21d = returns.rolling(window=21).std() * np.sqrt(252)
    
    # Volatilidades dos Modelos (Nível 3)
    print("  Estimando Heston e GARCH...")
    heston_vol = estimate_heston_vol(returns, ticker=TICKER, verbose=False)
    garch_vol = estimate_garch_vol(returns, variant="GARCH")
    
    # Alinhamento
    idx = prices.index.intersection(heston_vol.index).intersection(garch_vol.index).intersection(vol_realized_21d.index)
    prices = prices.reindex(idx)
    returns = returns.reindex(idx)
    heston_vol = heston_vol.reindex(idx)
    garch_vol = garch_vol.reindex(idx)
    vol_realized_21d = vol_realized_21d.reindex(idx)
    
    cfg = SignalConfig(profile=PROFILE)
    
    # 2. Execução dos 4 Cenários
    print("\n[2/4] Executando Backtests dos 4 Níveis...")
    
    # Nível 1: EMA Only (Base)
    print("  → Nível 1: EMA Only...")
    sig_lvl1 = build_ema_only_signals(prices, ema_fast=cfg.ema_fast, ema_slow=cfg.ema_slow)
    
    # Preparação Explícita de Inputs para Nível 2 e 4
    print("\n  [DEBUG] Preparando inputs para Nível 2 e 4...")
    # Nível 2: Shift(1) -> Vol de Ontem
    vol_input_lvl2 = vol_realized_21d.shift(1).fillna(method='bfill')
    
    # Nível 4: Shift(-1) -> Vol de Amanhã
    vol_input_lvl4 = vol_realized_21d.shift(-1).fillna(method='ffill')
    
    # Sanity Check - Dump to file
    with open("debug_inputs.txt", "w") as f:
        f.write("SANITY CHECK INPUTS\n")
        f.write("="*40 + "\n")
        f.write(f"Lvl2 (Retrovisor) Head:\n{vol_input_lvl2.head(10).to_string()}\n\n")
        f.write(f"Lvl4 (Oráculo) Head:\n{vol_input_lvl4.head(10).to_string()}\n\n")
        f.write(f"Are they equal? {vol_input_lvl2.equals(vol_input_lvl4)}\n")
        f.write(f"Correlation: {vol_input_lvl2.corr(vol_input_lvl4)}\n")
        
    if vol_input_lvl2.equals(vol_input_lvl4):
        print("  [ERRO CRÍTICO] Inputs do Nível 2 e 4 são IDÊNTICOS! Verifique a lógica de shift.")
    else:
        print("  [OK] Inputs são diferentes.")

    # Nível 2: EMA + Volatilidade Padrão (Retrovisor)
    print("  → Nível 2: Retrovisor (Vol Histórica)...")
    sig_lvl2 = build_signals(prices, garch_vol=vol_input_lvl2, heston_vol=vol_input_lvl2, cfg=cfg)
    
    # Nível 3: Meta-Estratégia (Nosso Modelo)
    print("  → Nível 3: Meta-Estratégia (Heston + GARCH)...")
    sig_lvl3 = build_signals(prices, garch_vol=garch_vol, heston_vol=heston_vol, cfg=cfg)
    
    # Nível 4: Oráculo (Limite Teórico)
    print("  → Nível 4: Oráculo (Vol Futura)...")
    sig_lvl4 = build_signals(prices, garch_vol=vol_input_lvl4, heston_vol=vol_input_lvl4, cfg=cfg)
    
    # 3. Cálculo de Performance
    print("\n[3/4] Calculando Métricas...")
    
    results = {}
    scenarios = [
        ("Nível 1 (EMA Only)", sig_lvl1, "gray", ":", 1.5),
        ("Nível 2 (Retrovisor)", sig_lvl2, "skyblue", "-", 1.5),
        ("Nível 3 (Meta-Estratégia)", sig_lvl3, "teal", "-", 2.5), # Destaque
        ("Nível 4 (Oráculo)", sig_lvl4, "gold", "--", 1.5)
    ]
    
    metrics_data = []
    
    plt.figure(figsize=(12, 7))
    
    for name, sig, color, style, lw in scenarios:
        strat_ret = _strategy_returns(sig)
        equity = _equity_curve(strat_ret)
        
        # Alinhar equity para começar em 1.0 no mesmo ponto (interseção comum)
        # Como todas usam os mesmos dados base alinhados, já devem estar ok, mas garantindo:
        equity = equity.reindex(idx).fillna(method='ffill').fillna(1.0)
        equity = equity / equity.iloc[0] # Normalizar base 1.0
        
        total_ret = equity.iloc[-1] - 1.0
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() > 0 else 0
        dd = _drawdown(equity).min()
        
        results[name] = {
            "Total Return": total_ret,
            "Sharpe": sharpe,
            "Max DD": dd,
            "Equity": equity
        }
        
        metrics_data.append({
            "Nível": name,
            "Sharpe": f"{sharpe:.2f}",
            "Retorno Total": f"{total_ret:.2%}",
            "Max Drawdown": f"{dd:.2%}"
        })
        
        plt.plot(equity.index, equity, label=f"{name} (Sharpe: {sharpe:.2f})", color=color, linestyle=style, linewidth=lw)

    # 4. Visualização
    print("\n[4/4] Gerando Gráfico e Relatório...")
    
    plt.title(f"Evolução da Estratégia: O Valor da Previsão de Volatilidade ({TICKER})", fontsize=14)
    plt.ylabel("Capital Acumulado (Base 1.0)")
    plt.xlabel("Data")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Tabela de Métricas no Gráfico (Opcional, mas pedido na legenda, já incluído na legenda acima)
    # Salvando
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True, parents=True)
    out_png = outdir / "evolution_chart.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"  ✓ Gráfico salvo em: {out_png}")
    # plt.show() # Não mostrar interativamente para não bloquear script se rodado em background
    
    # Relatório Final
    report_path = outdir / "evolution_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TABELA DE PERFORMANCE COMPARATIVA\n")
        f.write("=" * 80 + "\n")
        df_metrics = pd.DataFrame(metrics_data)
        f.write(df_metrics.to_string(index=False) + "\n")
        f.write("-" * 80 + "\n")
        
        # Atribuição de Valor
        sharpe_lvl2 = results["Nível 2 (Retrovisor)"]["Sharpe"]
        sharpe_lvl3 = results["Nível 3 (Meta-Estratégia)"]["Sharpe"]
        
        if sharpe_lvl2 > 0:
            gain_pct = ((sharpe_lvl3 - sharpe_lvl2) / sharpe_lvl2) * 100
            f.write("\nATRIBUIÇÃO DE VALOR (Previsão Matemática):\n")
            f.write(f"Ganho de Sharpe do Nível 3 sobre o Nível 2: +{gain_pct:.1f}%\n")
            f.write("Isso representa o valor adicionado por usar modelos Heston/GARCH vs Volatilidade Passada.\n")
        else:
            f.write("\nATRIBUIÇÃO DE VALOR:\n")
            f.write(f"Sharpe Nível 2: {sharpe_lvl2:.2f}\n")
            f.write(f"Sharpe Nível 3: {sharpe_lvl3:.2f}\n")
            f.write("Comparação direta prejudicada por Sharpe negativo ou zero no Nível 2.\n")

        f.write("=" * 80 + "\n")
    
    print(f"  ✓ Relatório salvo em: {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_evolution_analysis()

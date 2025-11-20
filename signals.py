# signals.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np
import pandas as pd

Profile = Literal["conservative", "moderate", "aggressive"]


@dataclass(frozen=True)
class SignalConfig:
    vol_hist_window: int = 7
    zscore_window: int = 30
    ema_fast: int = 7
    ema_slow: int = 21
    profile: Profile = "moderate"
    z_thresholds: dict | None = None
    # ema_fast_min_slope: float = 0.0005  # Inclinação mínima da EMA rápida (0.05% do preço) - DESATIVADO

    def __post_init__(self):
        if object.__getattribute__(self, "z_thresholds") is None:
            # Thresholds para COMPRA (negativos) e VENDA (positivos)
            object.__setattr__(
                self,
                "z_thresholds",
                {
                    "conservative": {"buy": -2.0, "sell": +2.0},
                    "moderate": {"buy": -1.0, "sell": +1.0},
                    "aggressive": {"buy": -0.5, "sell": +0.5},
                },
            )


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _align_series(*series: pd.Series) -> Tuple[pd.Index, Tuple[pd.Series, ...]]:
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    idx = idx.sort_values()
    return idx, tuple(s.reindex(idx) for s in series)


def _as_series(x: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    """Ensure we are always working with a Series, not a single-column DataFrame."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            s = x.iloc[:, 0].copy()
        else:
            raise ValueError(f"DataFrame for {name} has no columns.")
    else:
        s = x.copy()
    s.name = name
    return s


def build_signals(
    prices: pd.Series,
    garch_vol: pd.Series,
    heston_vol: pd.Series,
    cfg: SignalConfig = SignalConfig(),
) -> pd.DataFrame:
    """
    Meta-Estratégia Assimétrica 2.0:
    
    1. Engenharia de Features:
       - Vol_Benchmark: Média móvel de 7 dias da Vol Realizada de 21 dias
       - Vol_StdDev: Desvio padrão móvel (7 dias) da Volatilidade Histórica
       - Vol_Forecast_Final: Consenso inteligente (Regras A, B, C)
       - Z-Score: (Vol_Forecast_Final - Vol_Benchmark) / Vol_StdDev
    
    2. Lógica de Entrada Assimétrica:
       - COMPRA: Gatilho (EMA 7 cruza acima EMA 21) + Filtro (Z < -0.5) + OVERRIDE (divergência > 1%)
       - VENDA: Gatilho (EMA 7 cruza abaixo EMA 21) + Filtro rigoroso (Z > 1.5)
    
    3. Lógica de Saída:
       - Long Exit: EMA 7 cruza abaixo EMA 21
       - Short Exit: EMA 7 cruza acima EMA 21 OU Z-Score < 0.5
    """
    # 0) Align inputs and squeeze to Series
    idx, (p_in, gv_in, hv_in) = _align_series(prices, garch_vol, heston_vol)
    p = _as_series(p_in, "price")
    gv_in = _as_series(gv_in, "garch_vol")
    hv_in = _as_series(hv_in, "heston_vol")

    # 1) Returns and realized vol benchmark
    ret = p.pct_change()
    ret.name = "returns"

    # ============ 1. ENGENHARIA DE FEATURES ============
    
    # Step 1: Volatilidade realizada de 21 dias (anualizada)
    vol_realized_21d = ret.rolling(window=21, min_periods=21).std() * np.sqrt(252)
    
    # Step 2: Vol_Benchmark = Média móvel de 7 dias da Vol Realizada de 21 dias
    vol_benchmark = vol_realized_21d.rolling(window=7, min_periods=7).mean()
    vol_benchmark.name = "vol_benchmark"
    
    # Step 3: Vol_StdDev = Desvio padrão móvel (7 dias) da Volatilidade Histórica
    vol_stddev = vol_realized_21d.rolling(window=7, min_periods=7).std()
    vol_stddev.name = "vol_stddev"
    
    # Step 4: Vol_Forecast_Final (Consenso Inteligente)
    # Regra A (Concordância): Se ambos concordam, use média simples
    # Regra B (GARCH Wins): Se Heston > Benchmark mas GARCH < Benchmark, use GARCH
    # Regra C (Incerteza): Se Heston < Benchmark mas GARCH > Benchmark, force alto (Risco)
    
    below_bench_g = gv_in < vol_benchmark
    below_bench_h = hv_in < vol_benchmark
    above_bench_g = gv_in > vol_benchmark
    above_bench_h = hv_in > vol_benchmark
    
    # Regra A: Concordância (ambos > Benchmark ou ambos < Benchmark)
    agree_both_below = below_bench_g & below_bench_h
    agree_both_above = above_bench_g & above_bench_h
    agree_flag = agree_both_below | agree_both_above
    
    # Regra B: GARCH Wins (Heston > Benchmark mas GARCH < Benchmark)
    garch_wins = above_bench_h & below_bench_g
    
    # Regra C: Incerteza (Heston < Benchmark mas GARCH > Benchmark) → Force alto (Risco)
    uncertainty = below_bench_h & above_bench_g
    
    # Aplicar regras
    vol_forecast_final = pd.Series(index=idx, dtype=float, name="vol_forecast_final")
    
    # Regra A: Média simples quando concordam
    vol_forecast_final[agree_flag] = (gv_in[agree_flag] + hv_in[agree_flag]) / 2.0
    
    # Regra B: Use GARCH quando GARCH Wins
    vol_forecast_final[garch_wins] = gv_in[garch_wins]
    
    # Regra C: Force alto (Risco) - use o maior valor ou um múltiplo do benchmark
    # Usamos o máximo entre Heston e GARCH, ou 1.5x o benchmark (o que for maior)
    vol_forecast_final[uncertainty] = np.maximum(
        np.maximum(gv_in[uncertainty], hv_in[uncertainty]),
        vol_benchmark[uncertainty] * 1.5
    )
    
    # Preencher valores faltantes com média simples (fallback)
    vol_forecast_final = vol_forecast_final.fillna((gv_in + hv_in) / 2.0)
    
    # Risk state para análise
    risk_state = pd.Series("Neutral", index=idx, dtype=object)
    risk_state.name = "risk_state"
    risk_state[agree_both_below] = "Calmaria"
    risk_state[agree_both_above] = "Risco"
    risk_state[garch_wins] = "Calmaria (GARCH Wins)"
    risk_state[uncertainty] = "Risco (Incerteza)"
    
    # Step 5: Z-Score usando Vol_Forecast_Final
    # Z-Score = (Vol_Forecast_Final - Vol_Benchmark) / Vol_StdDev
    z = (vol_forecast_final - vol_benchmark) / vol_stddev.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan)
    z.name = "zscore"
    
    # Manter vol_pred_cons para compatibilidade (média simples)
    vol_pred_cons = (gv_in + hv_in) / 2.0
    vol_pred_cons.name = "vol_pred_cons"

    # ============ 2. EMAs E ANÁLISE DE TENDÊNCIA ============
    ema_fast = _ema(p, cfg.ema_fast)
    ema_fast.name = f"ema{cfg.ema_fast}"
    ema_slow = _ema(p, cfg.ema_slow)
    ema_slow.name = f"ema{cfg.ema_slow}"
    
    # Trend direction
    trend_up = ema_fast > ema_slow
    trend_down = ema_fast < ema_slow
    trend_up.name = "trend_up"
    trend_down.name = "trend_down"
    
    # Detectar cruzamentos (gatilhos)
    trend_up_prev = trend_up.shift(1).fillna(False)
    trend_down_prev = trend_down.shift(1).fillna(False)
    
    # Gatilho COMPRA: EMA 7 cruza acima EMA 21
    ema_cross_up = trend_up & (~trend_up_prev)
    ema_cross_up.name = "ema_cross_up"
    
    # Gatilho VENDA: EMA 7 cruza abaixo EMA 21
    ema_cross_down = trend_down & (~trend_down_prev)
    ema_cross_down.name = "ema_cross_down"
    
    # Divergência entre EMAs (para OVERRIDE)
    # Divergência = (EMA_fast - EMA_slow) / EMA_slow (em %)
    ema_divergence_pct = ((ema_fast - ema_slow) / ema_slow.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ema_divergence_pct.name = "ema_divergence_pct"
    
    # EMA momentum (para análise)
    ema_fast_diff = ema_fast.diff()
    ema_slow_diff = ema_slow.diff()
    ema_fast_diff.name = "ema_fast_diff"
    ema_slow_diff.name = "ema_slow_diff"
    
    # ============ 3. LÓGICA DE ENTRADA ASSIMÉTRICA ============
    
    # COMPRA (Long):
    # Gatilho: EMA 7 cruza acima EMA 21
    # Filtro: Z-Score < -0.5 (Calmaria)
    # OVERRIDE: Se divergência > 1%, ignore filtro até Z-Score +1.0
    
    buy_vol_filter = (z < -0.5)  # Filtro padrão: Calmaria
    buy_override = (ema_divergence_pct > 0.01) & (z < 1.0)  # OVERRIDE: divergência > 1% e Z < 1.0
    buy_signal = ema_cross_up & (buy_vol_filter | buy_override)
    buy_signal.name = "buy_signal"
    
    # VENDA (Short):
    # Gatilho: EMA 7 cruza abaixo EMA 21
    # Filtro rigoroso: Z-Score > 1.5 (Medo Real)
    
    sell_vol_filter = (z > 1.5)  # Filtro rigoroso: Medo Real
    sell_signal = ema_cross_down & sell_vol_filter
    sell_signal.name = "sell_signal"
    
    # Flags para análise
    buy_gate = buy_vol_filter
    buy_gate.name = "buy_gate"
    sell_gate = sell_vol_filter
    sell_gate.name = "sell_gate"
    
    # Flags de confluência/divergência (mantidas para compatibilidade)
    ema_fast_rising = ema_fast_diff > 0
    ema_slow_rising = ema_slow_diff > 0
    ema_fast_falling = ema_fast_diff < 0
    ema_slow_falling = ema_slow_diff < 0
    ema_confluent_buy = ema_fast_rising & ema_slow_rising
    ema_confluent_sell = ema_fast_falling & ema_slow_falling
    ema_divergent = (ema_fast_rising & ema_slow_falling) | (ema_fast_falling & ema_slow_rising)
    ema_confluent_buy.name = "ema_confluent_buy"
    ema_confluent_sell.name = "ema_confluent_sell"
    ema_divergent.name = "ema_divergent"
    
    # EMA slope (para análise)
    ema_fast_slope = (ema_fast_diff / p.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ema_fast_slope.name = "ema_fast_slope"
    ema_fast_strong_up = pd.Series(True, index=idx, name="ema_fast_strong_up")
    ema_fast_strong_down = pd.Series(True, index=idx, name="ema_fast_strong_down")
    
    # ============ BACKUP DA LÓGICA COM FILTRO DE INCLINAÇÃO ============
    # LÓGICA COM INCLINAÇÃO: Confluência/divergência + Filtro de Inclinação Mínima
    # A EMA rápida precisa ter inclinação suficiente para mostrar FORÇA na direção
    # if cfg.profile == "aggressive":
    #     buy_ema_confluence = ema_confluent_buy | (ema_divergent & trend_up)
    # else:
    #     buy_ema_confluence = ema_confluent_buy
    # buy_ema_filter_with_slope = buy_ema_confluence & ema_fast_strong_up
    # buy_signal_with_slope = buy_gate & trend_up & buy_ema_filter_with_slope
    # 
    # if cfg.profile == "aggressive":
    #     sell_ema_confluence = ema_confluent_sell | (ema_divergent & trend_down)
    # else:
    #     sell_ema_confluence = ema_confluent_sell
    # sell_ema_filter_with_slope = sell_ema_confluence & ema_fast_strong_down
    # sell_signal_with_slope = sell_gate & trend_down & sell_ema_filter_with_slope
    # ===============================================================================

    # ============ 4. LÓGICA DE SAÍDA ASSIMÉTRICA ============
    # Position: +1 (compra), -1 (venda), 0 (neutro)
    position = pd.Series(0, index=idx, dtype=int, name="position")
    
    # Long Exit: Quando EMA 7 cruza abaixo EMA 21
    # Short Exit: Quando EMA 7 cruza acima EMA 21 OU Z-Score < 0.5 (Saída de Pânico)
    
    for i in range(1, len(idx)):
        prev_pos = position.iat[i - 1]
        
        if prev_pos == 1:  # Em posição de COMPRA (Long)
            # Long Exit: EMA 7 cruza abaixo EMA 21
            if trend_down.iat[i]:
                position.iat[i] = 0  # Sair
            else:
                position.iat[i] = 1  # Manter
                
        elif prev_pos == -1:  # Em posição de VENDA (Short)
            # Short Exit: EMA 7 cruza acima EMA 21 OU Z-Score < 0.5
            z_val = z.iat[i] if not pd.isna(z.iat[i]) else np.inf
            if trend_up.iat[i] or (z_val < 0.5):
                position.iat[i] = 0  # Sair
            else:
                position.iat[i] = -1  # Manter
                
        else:  # Neutro (0)
            # Entra em compra ou venda baseado nos sinais
            if buy_signal.iat[i]:
                position.iat[i] = 1
            elif sell_signal.iat[i]:
                position.iat[i] = -1
            else:
                position.iat[i] = 0
    
    # ============ BACKUP DA LÓGICA ALTERNATIVA (comentado) ============
    # LÓGICA ALTERNATIVA: Saída baseada na mudança de direção da EMA rápida
    # Saída mais sensível: quando a média curta (EMA rápida) vira contra a tendência
    # 
    # REGRAS DE SAÍDA ALTERNATIVA:
    # - LONG: Sai quando EMA rápida começa a DESCER (ema_fast_diff < 0)
    # - SHORT: Sai quando EMA rápida começa a SUBIR (ema_fast_diff > 0)
    # 
    # VANTAGEM: Mais rápido em detectar reversões, captura movimentos mais cedo
    # DESVANTAGEM: Pode sair muito cedo em movimentos laterais
    # 
    # for i in range(1, len(idx)):
    #     prev_pos = position.iat[i - 1]
    #     
    #     if prev_pos == 1:  # Em posição de COMPRA (LONG)
    #         # Sai quando EMA rápida começa a descer (virar contra)
    #         if pd.isna(ema_fast_diff.iat[i]):
    #             position.iat[i] = prev_pos
    #         elif ema_fast_diff.iat[i] < 0:
    #             # EMA rápida descendo → SAIR da posição LONG
    #             position.iat[i] = 0
    #         else:
    #             # EMA rápida subindo ou estável → MANTER posição LONG
    #             position.iat[i] = 1
    #             
    #     elif prev_pos == -1:  # Em posição de VENDA (SHORT)
    #         # Sai quando EMA rápida começa a subir (virar contra)
    #         if pd.isna(ema_fast_diff.iat[i]):
    #             position.iat[i] = prev_pos
    #         elif ema_fast_diff.iat[i] > 0:
    #             # EMA rápida subindo → SAIR da posição SHORT
    #             position.iat[i] = 0
    #         else:
    #             # EMA rápida descendo ou estável → MANTER posição SHORT
    #             position.iat[i] = -1
    #             
    #     else:  # Neutro (0)
    #         if buy_signal.iat[i]:
    #             position.iat[i] = 1
    #         elif sell_signal.iat[i]:
    #             position.iat[i] = -1
    #         else:
    #             position.iat[i] = 0
    # =================================================================

    # ============ 5. ASSEMBLE DATAFRAME ============
    df = pd.concat(
        [
            p,
            ret,
            ema_fast,
            ema_slow,
            ema_fast_diff,
            ema_slow_diff,
            ema_divergence_pct,
            vol_benchmark,
            vol_stddev,
            vol_forecast_final,
            vol_pred_cons,  # Mantido para compatibilidade
            gv_in,
            hv_in,
            z,
            risk_state,
            agree_flag,
            buy_gate,
            sell_gate,
            trend_up,
            trend_down,
            ema_cross_up,
            ema_cross_down,
            ema_confluent_buy,
            ema_confluent_sell,
            ema_divergent,
            ema_fast_slope,
            ema_fast_strong_up,
            ema_fast_strong_down,
            buy_signal,
            sell_signal,
            position,
        ],
        axis=1,
    )

    # Warm-up: disable early positions (onde não há dados suficientes)
    warmup_mask = vol_benchmark.isna() | z.isna() | ema_fast.isna() | ema_slow.isna()
    df.loc[warmup_mask, "position"] = 0
    df["position"] = df["position"].astype(int)

    return df


def build_ema_only_signals(prices: pd.Series, ema_fast: int = 7, ema_slow: int = 21) -> pd.DataFrame:
    """
    Estratégia simples baseada APENAS em cruzamento de médias móveis exponenciais.
    Usado para comparar com a meta-estratégia que usa modelo de volatilidade.
    
    Regras:
    - Entrada LONG: quando EMA_fast cruza acima de EMA_slow (trend_up)
    - Entrada SHORT: quando EMA_fast cruza abaixo de EMA_slow (trend_down)
    - Saída: quando o cruzamento inverte (trend_up vira trend_down ou vice-versa)
    
    Args:
        prices: Série de preços
        ema_fast: Período da EMA rápida (padrão: 7)
        ema_slow: Período da EMA lenta (padrão: 21)
    
    Returns:
        DataFrame com preços, retornos, EMAs, sinais e posições
    """
    idx = prices.index
    p = prices.copy()
    p.name = "price"
    
    # Retornos
    ret = p.pct_change().dropna()
    ret.name = "returns"
    
    # Alinhar índices
    idx, (p, ret) = _align_series(p, ret)
    
    # Calcular EMAs
    ema_fast_series = _ema(p, span=ema_fast)
    ema_fast_series.name = f"ema{ema_fast}"
    ema_slow_series = _ema(p, span=ema_slow)
    ema_slow_series.name = f"ema{ema_slow}"
    
    # Tendência: EMA rápida vs EMA lenta
    trend_up = (ema_fast_series > ema_slow_series)
    trend_up.name = "trend_up"
    trend_down = (ema_fast_series < ema_slow_series)
    trend_down.name = "trend_down"
    
    # Sinais de entrada baseados em cruzamento
    # Entrada LONG: quando trend_up se torna True (cruzamento para cima)
    # Entrada SHORT: quando trend_down se torna True (cruzamento para baixo)
    trend_up_prev = trend_up.shift(1).fillna(False)
    trend_down_prev = trend_down.shift(1).fillna(False)
    
    # Buy signal: cruzamento para cima (trend_up novo)
    buy_signal = trend_up & (~trend_up_prev)
    buy_signal.name = "buy_signal"
    
    # Sell signal: cruzamento para baixo (trend_down novo)
    sell_signal = trend_down & (~trend_down_prev)
    sell_signal.name = "sell_signal"
    
    # Lógica de posição (swing trade)
    position = pd.Series(0, index=idx, dtype=int, name="position")
    
    for i in range(1, len(idx)):
        prev_pos = position.iat[i - 1]
        
        if prev_pos == 1:  # Em posição de COMPRA
            # Mantém compra enquanto trend_up, fecha se trend_down
            position.iat[i] = 1 if trend_up.iat[i] else 0
        elif prev_pos == -1:  # Em posição de VENDA
            # Mantém venda enquanto trend_down, fecha se trend_up
            position.iat[i] = -1 if trend_down.iat[i] else 0
        else:  # Neutro (0)
            # Entra em compra ou venda baseado nos sinais de cruzamento
            if buy_signal.iat[i]:
                position.iat[i] = 1
            elif sell_signal.iat[i]:
                position.iat[i] = -1
            else:
                position.iat[i] = 0
    
    # Assemble DataFrame
    df = pd.concat(
        [
            p,
            ret,
            ema_fast_series,
            ema_slow_series,
            trend_up,
            trend_down,
            buy_signal,
            sell_signal,
            position,
        ],
        axis=1,
    )
    
    # Warm-up: disable early positions (onde não há dados suficientes para EMAs)
    warmup_mask = ema_fast_series.isna() | ema_slow_series.isna()
    df.loc[warmup_mask, "position"] = 0
    df["position"] = df["position"].astype(int)
    
    return df

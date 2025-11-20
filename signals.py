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
    Implements the meta-strategy logic:
      1) Consensus predicted volatility (GARCH + Heston)/2
      2) Benchmark: Média Móvel de 7 dias da Vol Realizada de 21 dias
      3) Z-Score of predicted vol vs benchmark vol
      4) COMPRA: Z-Score < threshold (Calmaria) + EMAs para cima
      5) VENDA: Z-Score > threshold (Risco) + EMAs para baixo
      6) Discriminação por perfil baseada na confluência das EMAs
      7) Swing-trade logic: exit only on trend reversal
    """
    # 0) Align inputs and squeeze to Series
    idx, (p_in, gv_in, hv_in) = _align_series(prices, garch_vol, heston_vol)
    p = _as_series(p_in, "price")
    gv_in = _as_series(gv_in, "garch_vol")
    hv_in = _as_series(hv_in, "heston_vol")

    # 1) Returns and realized vol benchmark
    ret = p.pct_change()
    ret.name = "returns"

    # Benchmark: Média Móvel de 7 dias da Vol Realizada de 21 dias
    # Step 1: Volatilidade realizada de 21 dias
    vol_realized_21d = ret.rolling(window=21, min_periods=21).std()
    # Step 2: Média móvel de 7 dias dessa volatilidade
    vol_hist = vol_realized_21d.rolling(window=7, min_periods=7).mean()
    vol_hist.name = "vol_hist_benchmark"  # Benchmark de volatilidade

    # 2) Consensus predicted vol
    vol_pred_cons = (gv_in + hv_in) / 2.0
    vol_pred_cons.name = "vol_pred_cons"

    # 3) Agreement flags (Calmaria / Risco / Neutral)
    below_hist_g = gv_in < vol_hist
    below_hist_h = hv_in < vol_hist
    agree_flag = (below_hist_g & below_hist_h) | ((~below_hist_g) & (~below_hist_h))
    agree_flag.name = "agree_flag"

    risk_state = pd.Series("Neutral", index=idx, dtype=object)
    risk_state.name = "risk_state"
    risk_state[below_hist_g & below_hist_h] = "Calmaria"
    risk_state[(~below_hist_g) & (~below_hist_h)] = "Risco"

    # 4) Z-Score of predicted vol vs benchmark vol
    # Z-Score = (Previsão_Consenso - Benchmark_Vol) / Benchmark_StdDev
    mu = vol_hist.rolling(cfg.zscore_window).mean()  # Média do benchmark
    sd = vol_hist.rolling(cfg.zscore_window).std()     # Desvio padrão do benchmark
    z = (vol_pred_cons - mu) / sd.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan)
    z.name = "zscore"

    # 5) EMAs and trend analysis
    ema_fast = _ema(p, cfg.ema_fast)
    ema_fast.name = f"ema{cfg.ema_fast}"
    ema_slow = _ema(p, cfg.ema_slow)
    ema_slow.name = f"ema{cfg.ema_slow}"
    
    # Trend direction
    trend_up = ema_fast > ema_slow
    trend_down = ema_fast < ema_slow
    trend_up.name = "trend_up"
    trend_down.name = "trend_down"
    
    # EMA momentum (direção das EMAs)
    ema_fast_diff = ema_fast.diff()  # Mudança absoluta na EMA rápida
    ema_slow_diff = ema_slow.diff()  # Mudança absoluta na EMA lenta
    
    # Inclinação relativa da EMA rápida (normalizada pelo preço)
    # Isso permite comparar a força do movimento independente do nível de preço
    # Inclinação = mudança absoluta / preço atual (em %)
    ema_fast_slope = (ema_fast_diff / p.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ema_fast_slope.name = "ema_fast_slope"
    
    # Confluência: ambas EMAs na mesma direção
    # Para COMPRA: ambas subindo (confluentes) OU divergentes (só agressivo)
    ema_fast_rising = ema_fast_diff > 0  # EMA rápida subindo
    ema_slow_rising = ema_slow_diff > 0  # EMA lenta subindo
    ema_fast_falling = ema_fast_diff < 0  # EMA rápida descendo
    ema_slow_falling = ema_slow_diff < 0  # EMA lenta descendo
    
    # Confluência para COMPRA: ambas subindo
    ema_confluent_buy = ema_fast_rising & ema_slow_rising
    # Confluência para VENDA: ambas descendo
    ema_confluent_sell = ema_fast_falling & ema_slow_falling
    # Divergência: direções opostas (só agressivo)
    ema_divergent = (ema_fast_rising & ema_slow_falling) | (ema_fast_falling & ema_slow_rising)
    
    ema_confluent_buy.name = "ema_confluent_buy"
    ema_confluent_sell.name = "ema_confluent_sell"
    ema_divergent.name = "ema_divergent"
    
    # Filtro de inclinação mínima (força da EMA rápida) - DESATIVADO
    # Mantido apenas para análise/debugging, mas não usado na lógica de entrada
    # Para COMPRA: EMA rápida deve ter inclinação positiva acima do threshold
    # ema_fast_min_slope = 0.0005  # 0.05% do preço
    # ema_fast_strong_up = ema_fast_slope > ema_fast_min_slope
    # Para VENDA: EMA rápida deve ter inclinação negativa abaixo do threshold negativo
    # ema_fast_strong_down = ema_fast_slope < -ema_fast_min_slope
    # 
    # ema_fast_strong_up.name = "ema_fast_strong_up"
    # ema_fast_strong_down.name = "ema_fast_strong_down"
    
    # Criando flags vazias para manter compatibilidade (mas não usadas na lógica)
    ema_fast_strong_up = pd.Series(True, index=idx, name="ema_fast_strong_up")  # Sempre True (desativado)
    ema_fast_strong_down = pd.Series(True, index=idx, name="ema_fast_strong_down")  # Sempre True (desativado)

    # 6) Profile gates for BUY and SELL
    z_thresh = cfg.z_thresholds[cfg.profile]
    buy_threshold = z_thresh["buy"]  # Negativo (ex: -1.0)
    sell_threshold = z_thresh["sell"]  # Positivo (ex: +1.0)
    
    # Buy gate: Z-Score < threshold (Calmaria)
    buy_gate = (z < buy_threshold)
    buy_gate.name = "buy_gate"
    
    # Sell gate: Z-Score > threshold (Risco)
    sell_gate = (z > sell_threshold)
    sell_gate.name = "sell_gate"

    # 7) Entry signals with profile discrimination
    # REGRA CRÍTICA - NUNCA OPERAR CONTRA A TENDÊNCIA:
    # - COMPRA: Só se Calmaria (Z < threshold) + EMAs para CIMA (trend_up)
    # - VENDA: Só se Risco (Z > threshold) + EMAs para BAIXO (trend_down)
    # - Calmaria + EMAs para baixo → NINGUÉM entra (nem agressivo) - vai contra o mercado!
    # - Risco + EMAs para cima → NINGUÉM entra - vai contra o mercado!
    
    # COMPRA: Calmaria + EMAs para cima + confluência/divergência por perfil
    # OBRIGATÓRIO: trend_up (EMAs para cima) - SEM EXCEÇÃO!
    if cfg.profile == "aggressive":
        # Agressivo pode entrar em divergência, MAS APENAS se trend_up
        # Divergência permitida: EMA rápida subindo mas lenta descendo (ainda trend_up)
        # OU EMA rápida descendo mas lenta subindo (ainda trend_up) - mas isso é raro
        # Na prática: divergência só faz sentido se ainda estiver trend_up
        buy_ema_filter = ema_confluent_buy | (ema_divergent & trend_up)
    else:
        # Conservador/Moderado: só entra em confluência (ambas subindo)
        buy_ema_filter = ema_confluent_buy
    
    # COMPRA: buy_gate (Calmaria) + trend_up (EMAs para cima) + filtro de confluência
    # GARANTIA: Se trend_down, buy_signal = False (mesmo com Calmaria)
    buy_signal = buy_gate & trend_up & buy_ema_filter
    buy_signal.name = "buy_signal"
    
    # VENDA: Risco + EMAs para baixo + confluência/divergência por perfil
    # OBRIGATÓRIO: trend_down (EMAs para baixo) - SEM EXCEÇÃO!
    if cfg.profile == "aggressive":
        # Agressivo pode entrar em divergência, MAS APENAS se trend_down
        # Divergência permitida: EMA rápida descendo mas lenta subindo (ainda trend_down)
        # OU EMA rápida subindo mas lenta descendo (ainda trend_down) - mas isso é raro
        # Na prática: divergência só faz sentido se ainda estiver trend_down
        sell_ema_filter = ema_confluent_sell | (ema_divergent & trend_down)
    else:
        # Conservador/Moderado: só entra em confluência (ambas descendo)
        sell_ema_filter = ema_confluent_sell
    
    # VENDA: sell_gate (Risco) + trend_down (EMAs para baixo) + filtro de confluência
    # GARANTIA: Se trend_up, sell_signal = False (mesmo com Risco)
    sell_signal = sell_gate & trend_down & sell_ema_filter
    sell_signal.name = "sell_signal"
    
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

    # 8) Swing-trade persistence logic
    # Position: +1 (compra), -1 (venda), 0 (neutro)
    position = pd.Series(0, index=idx, dtype=int, name="position")
    
    # ============ LÓGICA ATIVA (RESTAURADA) ============
    # Saída baseada no cruzamento das EMAs (trend_up/trend_down)
    # Mantém posição enquanto a tendência permanece favorável
    for i in range(1, len(idx)):
        prev_pos = position.iat[i - 1]
        
        if prev_pos == 1:  # Em posição de COMPRA
            # Mantém compra enquanto trend_up, fecha se trend_down
            position.iat[i] = 1 if trend_up.iat[i] else 0
        elif prev_pos == -1:  # Em posição de VENDA
            # Mantém venda enquanto trend_down, fecha se trend_up
            position.iat[i] = -1 if trend_down.iat[i] else 0
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

    # 9) Assemble DataFrame
    # Adicionar ema_fast_diff e ema_slow_diff para análise da lógica de saída
    ema_fast_diff.name = "ema_fast_diff"
    ema_slow_diff.name = "ema_slow_diff"
    
    df = pd.concat(
        [
            p,
            ret,
            ema_fast,
            ema_slow,
            ema_fast_diff,  # Mudança na EMA rápida (usada para saída)
            ema_slow_diff,  # Mudança na EMA lenta (para análise)
            vol_hist,
            gv_in,
            hv_in,
            vol_pred_cons,
            z,
            risk_state,
            agree_flag,
            buy_gate,
            sell_gate,
            trend_up,
            trend_down,
            ema_confluent_buy,
            ema_confluent_sell,
            ema_divergent,
            ema_fast_slope,  # Inclinação relativa da EMA rápida
            ema_fast_strong_up,  # EMA rápida com inclinação forte para cima
            ema_fast_strong_down,  # EMA rápida com inclinação forte para baixo
            buy_signal,
            sell_signal,
            position,
        ],
        axis=1,
    )

    # Warm-up: disable early positions (onde não há dados suficientes)
    warmup_mask = vol_hist.isna() | z.isna() | ema_fast.isna() | ema_slow.isna()
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

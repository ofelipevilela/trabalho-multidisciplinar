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
    zscore_window: int = 21
    ema_fast: int = 7
    ema_slow: int = 21
    profile: Profile = "moderate"
    z_thresholds: dict | None = None
    
    # Novos parâmetros para Short Sniper
    vol_buffer: float = 0.05        # Margem de segurança de 5% acima do benchmark
    ema_slope_threshold: float = -0.0002  # Inclinação negativa mínima para gatilho de venda
    
    # Novo parâmetro para Reversão (Long)
    ema_slope_threshold_up: float = 0.0005 # Inclinação positiva forte para reversão
    panic_z_factor: float = 3.0            # Fator multiplicador para override de pânico

    def __post_init__(self):
        if object.__getattribute__(self, "z_thresholds") is None:
            # Padrões de Z-Score por Perfil de Risco
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
    Estratégia Pivotada: "Short Sniper" & "Long Simple" + Reversão por Inclinação
    
    1. Engenharia de Features:
       - Vol_Benchmark: Média móvel de 7 dias da Vol Realizada de 21 dias
       - Vol_Forecast_Final: Consenso inteligente (Regras A, B, C)
       - EMA Slope: Inclinação da EMA rápida e lenta
    
    2. Lógica de Entrada:
       - COMPRA (Long Simple): 
            a) Cruzamento de médias (EMA 7 > EMA 21)
            b) Reversão Forte: EMA 7 inclina forte pra cima (> threshold) E EMA 21 também sobe.
       - VENDA (Short Sniper): 
            (Vol_Forecast > Vol_Benchmark * (1 + buffer)) AND (EMA_Slope < threshold)
    
    3. Lógica de Saída:
       - Long Exit: EMA 7 cruza abaixo EMA 21
       - Short Exit (Panic Exit): 
            a) EMA 7 cruza acima EMA 21
            b) Volatilidade cai abaixo do Benchmark
            c) EMA 7 inclina forte pra cima (Reversão em V)
    """
    # 0) Align inputs and squeeze to Series
    idx, (p_in, gv_in, hv_in) = _align_series(prices, garch_vol, heston_vol)
    


    p = _as_series(p_in, "price")
    gv_in = _as_series(gv_in, "garch_vol")
    hv_in = _as_series(hv_in, "heston_vol")

    # Garantir que todas as volatilidades estejam na MESMA unidade (anualizada)
    TRADING_DAYS = 252
    scale = np.sqrt(TRADING_DAYS)
    gv_in = gv_in * scale
    hv_in = hv_in * scale

    # 1) Returns and realized vol benchmark
    ret = p.pct_change()
    ret.name = "returns"

    # ============ 1. ENGENHARIA DE FEATURES ============
    
    # Step 1: Volatilidade realizada de 21 dias (anualizada)
    vol_realized_21d = ret.rolling(window=21, min_periods=21).std() * np.sqrt(252)
    
    # Step 2: Vol_Benchmark = Média móvel de 7 dias da Vol Realizada de 21 dias
    vol_benchmark = vol_realized_21d.rolling(window=7, min_periods=7).mean()
    vol_benchmark.name = "vol_benchmark"
    
    # Step 3: Vol_StdDev (Mantido para Z-Score visual)
    vol_stddev = vol_realized_21d.rolling(window=7, min_periods=7).std()
    vol_stddev.name = "vol_stddev"
    
    # Step 4: Vol_Forecast_Final (Consenso Inteligente)
    below_bench_g = gv_in < vol_benchmark
    below_bench_h = hv_in < vol_benchmark
    above_bench_g = gv_in > vol_benchmark
    above_bench_h = hv_in > vol_benchmark
    
    agree_both_below = below_bench_g & below_bench_h
    agree_both_above = above_bench_g & above_bench_h
    agree_flag = agree_both_below | agree_both_above
    garch_wins = above_bench_h & below_bench_g
    uncertainty = below_bench_h & above_bench_g
    
    vol_forecast_final = pd.Series(index=idx, dtype=float, name="vol_forecast_final")
    vol_forecast_final[agree_flag] = (gv_in[agree_flag] + hv_in[agree_flag]) / 2.0
    vol_forecast_final[garch_wins] = gv_in[garch_wins]
    vol_forecast_final[uncertainty] = np.maximum(
        np.maximum(gv_in[uncertainty], hv_in[uncertainty]),
        vol_benchmark[uncertainty] * 1.5
    )
    vol_forecast_final = vol_forecast_final.fillna((gv_in + hv_in) / 2.0)
    
    # Step 5: Z-Score (Apenas para visualização/debug, não usado na lógica principal)
    min_stddev = vol_stddev.quantile(0.1)
    if pd.isna(min_stddev) or min_stddev <= 0:
        min_stddev = 0.005
    else:
        min_stddev = max(min_stddev, 0.005)
    
    vol_stddev_safe = vol_stddev.clip(lower=min_stddev)
    z = (vol_forecast_final - vol_benchmark) / vol_stddev_safe
    
    z = z.replace([np.inf, -np.inf], np.nan).clip(lower=-10.0, upper=10.0)
    z.name = "zscore"
    
    # Manter vol_pred_cons para compatibilidade
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
    
    # Gatilho VENDA: EMA 7 cruza abaixo EMA 21 (usado apenas como referência visual ou auxiliar)
    ema_cross_down = trend_down & (~trend_down_prev)
    ema_cross_down.name = "ema_cross_down"
    
    # EMA Slope (Inclinação)
    # Variação percentual de 1 dia
    ema_fast_slope = (ema_fast.diff() / p.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ema_fast_slope.name = "ema_fast_slope"
    
    ema_slow_slope = (ema_slow.diff() / p.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ema_slow_slope.name = "ema_slow_slope"
    
    # ============ 3. LÓGICA DE ENTRADA PIVOTADA ============
    
    # --- COMPRA (Long Simple + Reversão) ---
    # 1. Cruzamento de médias
    # 2. Reversão Forte: EMA 7 inclina forte pra cima (> threshold) E EMA 21 também sobe
    strong_reversal_up = (ema_fast_slope > cfg.ema_slope_threshold_up) & (ema_slow_slope > 0)
    
    buy_signal = ema_cross_up | strong_reversal_up
    buy_signal.name = "buy_signal"
    
    # --- VENDA (Short Sniper) ---
    # Regra: (Z-Score > Threshold do Perfil) AND (EMA_Slope < threshold)
    
    # Getting thresholds for current profile
    thresholds = cfg.z_thresholds[cfg.profile]
    sell_threshold = thresholds["sell"]
    
    # Condição 1: Z-Score Alto (Volatilidade anormal para o regime atual)
    # Isso substitui o buffer fixo de 5% por uma medida estatística de "Desvios Padrão"
    vol_condition = z > sell_threshold
    
    # Condição 2: Inclinação Negativa Acentuada
    slope_condition = ema_fast_slope < cfg.ema_slope_threshold
    
    # Condição 3 (NOVA): Pânico Extremo (Override)
    # Se o Z-Score for surreal (ex: > factor * threshold), vende IMEDIATAMENTE.
    # Isso permite que o Oráculo (Level 4) venda ANTES do preço cair.
    panic_override = z > (sell_threshold * cfg.panic_z_factor)
    
    # Sinal de Venda: (Vol Alta E Caindo) OU (Pânico Extremo)
    sell_signal = (vol_condition & slope_condition) | panic_override
    sell_signal.name = "sell_signal"
    
    # Flags auxiliares para visualização
    buy_gate = pd.Series(True, index=idx) # Sempre aberto para Long Simple
    buy_gate.name = "buy_gate"
    sell_gate = vol_condition # Gate de volatilidade para Short
    sell_gate.name = "sell_gate"

    # ============ 4. LÓGICA DE SAÍDA ============
    position = pd.Series(0, index=idx, dtype=int, name="position")
    
    for i in range(1, len(idx)):
        prev_pos = position.iat[i - 1]
        
        # Valores atuais
        is_trend_down = trend_down.iat[i]
        is_trend_up = trend_up.iat[i]
        current_vol = vol_forecast_final.iat[i] if not pd.isna(vol_forecast_final.iat[i]) else 0
        current_bench = vol_benchmark.iat[i] if not pd.isna(vol_benchmark.iat[i]) else 0
        current_fast_slope = ema_fast_slope.iat[i] if not pd.isna(ema_fast_slope.iat[i]) else 0
        
        # Estado atual da posição (pode mudar dentro do loop)
        curr_pos = prev_pos
        
        if curr_pos == 1:  # Em posição de COMPRA (Long)
            # Exit Long: Cruzamento para baixo (trend virou down)
            if is_trend_down:
                curr_pos = 0
            else:
                curr_pos = 1
                
        elif curr_pos == -1:  # Em posição de VENDA (Short)
            # Exit Short (Panic Exit):
            # 1. Reversão de tendência (trend virou up)
            # 2. Volatilidade caiu abaixo do benchmark (fim do pânico)
            # 3. Reversão Forte de Inclinação (V-Shape recovery)
            panic_over = current_vol < current_bench
            v_shape_reversal = current_fast_slope > cfg.ema_slope_threshold_up
            
            if is_trend_up or panic_over or v_shape_reversal:
                curr_pos = 0
            else:
                curr_pos = -1
        
        # Se estiver Neutro (ou acabou de sair), verificar entradas
        if curr_pos == 0:
            if buy_signal.iat[i]:
                curr_pos = 1
            elif sell_signal.iat[i]:
                curr_pos = -1
            else:
                curr_pos = 0
        
        position.iat[i] = curr_pos

    # ============ 5. ASSEMBLE DATAFRAME ============
    # Campos auxiliares para compatibilidade com visualização existente
    risk_state = pd.Series("Neutral", index=idx) # Placeholder
    
    df = pd.concat(
        [
            p,
            ret,
            ema_fast,
            ema_slow,
            ema_fast_slope,
            ema_slow_slope, # Adicionado
            vol_benchmark,
            vol_stddev,
            vol_forecast_final,
            vol_pred_cons,
            gv_in,
            hv_in,
            z,
            risk_state,
            buy_gate,
            sell_gate,
            trend_up,
            trend_down,
            ema_cross_up,
            ema_cross_down,
            buy_signal,
            sell_signal,
            position,
        ],
        axis=1,
    )

    # Warm-up
    warmup_mask = vol_benchmark.isna() | ema_fast.isna() | ema_slow.isna()
    df.loc[warmup_mask, "position"] = 0
    df["position"] = df["position"].astype(int)

    return df


def build_ema_only_signals(prices: pd.Series, ema_fast: int = 7, ema_slow: int = 21) -> pd.DataFrame:
    """
    Estratégia simples baseada APENAS em cruzamento de médias móveis exponenciais.
    Lógica: Stop-and-Reverse (Sempre posicionado)
    - EMA Rápida > EMA Lenta: COMPRA (+1)
    - EMA Rápida < EMA Lenta: VENDA (-1)
    """
    idx = prices.index
    p = prices.copy()
    p.name = "price"
    ret = p.pct_change()
    ret.name = "returns"
    # idx, (p, ret) = _align_series(p, ret)  <-- Causava o desalinhamento ao remover o primeiro item
    
    ema_fast_series = _ema(p, span=ema_fast)
    ema_fast_series.name = f"ema{ema_fast}"
    ema_slow_series = _ema(p, span=ema_slow)
    ema_slow_series.name = f"ema{ema_slow}"
    
    trend_up = (ema_fast_series > ema_slow_series)
    trend_up.name = "trend_up"
    trend_down = (ema_fast_series < ema_slow_series)
    trend_down.name = "trend_down"
    
    trend_up_prev = trend_up.shift(1).fillna(False)
    trend_down_prev = trend_down.shift(1).fillna(False)
    
    # Sinais apenas para registro (cruzamentos)
    buy_signal = trend_up & (~trend_up_prev)
    buy_signal.name = "buy_signal"
    sell_signal = trend_down & (~trend_down_prev)
    sell_signal.name = "sell_signal"
    
    # Posição baseada puramente na tendência (Stop and Reverse)
    position = pd.Series(0, index=idx, dtype=int, name="position")
    position[trend_up] = 1
    position[trend_down] = -1
    
    # Warm-up: zerar posições enquanto EMAs não são válidas
    warmup_mask = ema_fast_series.isna() | ema_slow_series.isna()
    position[warmup_mask] = 0
    
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
    
    return df
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

    # --- SMART VOLATILITY (Downside Filter) ---
    # Calculate Downside Volatility (std of negative returns)
    ret_neg = ret.copy()
    ret_neg[ret > 0] = 0
    vol_downside = ret_neg.rolling(window=21, min_periods=21).std()
    vol_downside.name = "vol_downside"
    
    # Z-Score of Downside Volatility (relative to its own history)
    mu_down = vol_downside.rolling(cfg.zscore_window).mean()
    sd_down = vol_downside.rolling(cfg.zscore_window).std()
    z_down = (vol_downside - mu_down) / sd_down.replace(0, np.nan)
    z_down = z_down.replace([np.inf, -np.inf], np.nan)
    z_down.name = "zscore_downside"

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
    ema_fast_slope = (ema_fast_diff / p.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ema_fast_slope.name = "ema_fast_slope"
    
    # Confluência: ambas EMAs na mesma direção
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
    
    ema_fast_strong_up = pd.Series(True, index=idx, name="ema_fast_strong_up")
    ema_fast_strong_down = pd.Series(True, index=idx, name="ema_fast_strong_down")

    # 6) Profile gates for BUY and SELL
    z_thresh = cfg.z_thresholds[cfg.profile]
    buy_threshold = z_thresh["buy"]  # Negativo (ex: -1.0)
    sell_threshold = z_thresh["sell"]  # Positivo (ex: +1.0)
    
    # Buy gate: SMART VOLATILITY -> Use Downside Z-Score instead of Total Z-Score
    # Se Downside Vol for baixa (Z < Threshold), permitimos a compra (mesmo que Vol Total seja alta)
    buy_gate = (z_down < 1.0) # Hardcoded threshold for safety (1.0 std dev above mean)
    # Note: Using 1.0 as a reasonable "Normal" limit. If > 1.0, downside risk is elevated.
    buy_gate.name = "buy_gate"
    
    # Sell gate: Keep original logic (High Total Vol is good for Shorting? Or High Downside?)
    # For Shorting, High Downside Vol is actually what we want to ride? 
    # Or do we want to enter BEFORE it crashes?
    # Let's keep original Sell Gate (High Total Vol) for now, as Shorting is risky.
    sell_gate = (z > sell_threshold)
    sell_gate.name = "sell_gate"

    # 7) Entry signals with profile discrimination
    # DYNAMIC EXIT STRATEGY:
    # 1. ALWAYS ENTER on Trend Up (Don't filter entries).
    # 2. LONG ONLY (Disable Shorts).
    
    # COMPRA: Apenas Trend Up (Médias Cruzadas para Cima)
    # Ignoramos buy_gate e filtros de volatilidade na entrada.
    buy_signal = trend_up
    buy_signal.name = "buy_signal"
    
    # VENDA: Desativada (Long-Only)
    # Apostar contra o S&P 500 tem sido perdedor.
    sell_signal = pd.Series(False, index=idx, name="sell_signal")
    
    # 8) Swing-trade persistence logic
    position = pd.Series(0, index=idx, dtype=int, name="position")
    
    # ============ LÓGICA ATIVA (DYNAMIC EXIT) ============
    # Regime Calmo (Z <= 0): Saída Lenta (Cruzamento de Médias)
    # Regime Risco (Z > 0):  Saída Rápida (2-Day Rule)
    
    days_against_trend = 0
    
    for i in range(1, len(idx)):
        prev_pos = position.iat[i - 1]
        curr_price = p.iat[i]
        curr_ema_fast = ema_fast.iat[i]
        curr_z_down = z_down.iat[i]
        
        # Tratar NaN no Z-Score (início da série) como Risco (modo seguro)
        # GENERALIZAÇÃO: Usar Z-Score de Downside em vez de Total
        # Isso evita sair de ralis fortes (alta volatilidade de alta)
        is_risky = True if pd.isna(curr_z_down) else (curr_z_down > 0)
        
        if prev_pos == 1:  # Em posição de COMPRA
            # Lógica de Saída Dinâmica
            should_exit = False
            
            if is_risky:
                # REGIME DE RISCO: Usar Saída Rápida (2-Day Rule)
                # Se preço fechar abaixo da EMA Rápida por 2 dias -> SAI
                if curr_price < curr_ema_fast:
                    days_against_trend += 1
                    if days_against_trend >= 2:
                        should_exit = True
                else:
                    days_against_trend = 0
                    
                # Também sai se a tendência virar (Cruzamento), caso o Fast Exit não tenha disparado
                if not trend_up.iat[i]:
                    should_exit = True
                    
            else:
                # REGIME CALMO: Usar Saída Lenta (Apenas Cruzamento)
                # Ignora "wicks" abaixo da EMA rápida. Só sai se a tendência virar.
                days_against_trend = 0 # Reset contador (não importa aqui)
                
                if not trend_up.iat[i]: # Cruzamento para baixo
                    should_exit = True
            
            # Aplica a decisão
            if should_exit:
                position.iat[i] = 0
                days_against_trend = 0
            else:
                position.iat[i] = 1
                
        elif prev_pos == -1:
            # Não deve acontecer em Long-Only, mas por segurança zeramos
            position.iat[i] = 0
                
        else:  # Neutro (0)
            days_against_trend = 0
            
            # Entra se houver sinal de compra (Trend Up)
            if buy_signal.iat[i]:
                position.iat[i] = 1
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
            vol_downside,  # New
            z,
            z_down,        # New
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

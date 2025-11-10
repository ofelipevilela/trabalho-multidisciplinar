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

    def __post_init__(self):
        if object.__getattribute__(self, "z_thresholds") is None:
            object.__setattr__(
                self,
                "z_thresholds",
                {
                    "conservative": -2.0,
                    "moderate": -1.0,
                    "aggressive": -0.2,
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
    Implements the slide logic (4→9):
      1) Consensus predicted volatility (GARCH + Heston)/2 with agreement check.
      2) Compare vol vs vol: predicted vs realized (7d rolling std of returns).
      3) Z-Score of predicted vol vs mu/sigma of realized vol (30d default).
      4) Profile gate on Z-Score (conservative/moderate/aggressive).
      5) Trend filter with EMA(7) > EMA(21).
      6) Swing-trade logic: enter only if both layers align; exit only on trend reversal.
    """
    # 0) Align inputs and squeeze to Series
    idx, (p_in, gv_in, hv_in) = _align_series(prices, garch_vol, heston_vol)
    p = _as_series(p_in, "price")
    gv_in = _as_series(gv_in, "garch_vol")
    hv_in = _as_series(hv_in, "heston_vol")

    # 1) Returns and realized vol benchmark
    ret = p.pct_change()
    ret.name = "returns"

    vol_hist = ret.rolling(cfg.vol_hist_window).std()
    vol_hist.name = f"vol_hist_{cfg.vol_hist_window}d"

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

    # 4) Z-Score of predicted vol vs stats of realized vol
    mu = vol_hist.rolling(cfg.zscore_window).mean()
    sd = vol_hist.rolling(cfg.zscore_window).std()
    z = (vol_pred_cons - mu) / sd.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan)
    z.name = "zscore"

    # 5) Profile gate (active only when Calmaria + agreement)
    z_thresh = cfg.z_thresholds[cfg.profile]
    profile_gate = (z < z_thresh)
    profile_gate.name = "profile_gate"

    # 6) Trend filter
    ema_fast = _ema(p, cfg.ema_fast)
    ema_fast.name = f"ema{cfg.ema_fast}"
    ema_slow = _ema(p, cfg.ema_slow)
    ema_slow.name = f"ema{cfg.ema_slow}"
    trend_up = ema_fast > ema_slow
    trend_up.name = "trend_up"

    # 7) Entry signal: Calmaria + agreement + profile_gate + trend_up
    calmaria_and_agree = (risk_state == "Calmaria") & agree_flag
    entry_signal = calmaria_and_agree & profile_gate & trend_up
    entry_signal.name = "entry_signal"

    # 8) Swing-trade persistence
    position = pd.Series(0, index=idx, dtype=int, name="position")
    for i in range(1, len(idx)):
        if position.iat[i - 1] == 1:
            position.iat[i] = 1 if trend_up.iat[i] else 0
        else:
            position.iat[i] = 1 if entry_signal.iat[i] else 0

    # 9) Assemble DataFrame
    df = pd.concat(
        [
            p,
            ret,
            ema_fast,
            ema_slow,
            vol_hist,
            gv_in,
            hv_in,
            vol_pred_cons,
            z,
            risk_state,
            agree_flag,
            profile_gate,
            trend_up,
            entry_signal,
            position,
        ],
        axis=1,
    )

    # Warm-up: disable early positions
    warmup_mask = vol_hist.isna() | z.isna() | ema_fast.isna() | ema_slow.isna()
    df.loc[warmup_mask, "position"] = 0
    df["position"] = df["position"].astype(int)

    return df

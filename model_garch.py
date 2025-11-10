# models/garch.py

from __future__ import annotations
from typing import Literal, Optional
import pandas as pd

try:
    from arch import arch_model
except Exception as e:
    raise ImportError(
        "The 'arch' package is required. Install it with:\n"
        "  python -m pip install arch"
    ) from e


GarchVariant = Literal["GARCH", "EGARCH", "GJR"]


def estimate_garch_vol(
    returns: pd.Series,
    variant: GarchVariant = "GARCH",
    dist: Literal["normal", "t", "skewt"] = "normal",
    p: int = 1,
    q: int = 1,
    o: Optional[int] = None,
    scale_100: bool = True,
) -> pd.Series:
    """
    Estimate conditional volatility using a (E)GARCH/GJR model and return a *single* series.

    This mirrors `estimate_heston_vol`: same input (returns) → same output shape (daily vol).
    No annualization, no comparisons, no VIX. Just the conditional volatility.

    Args:
        returns: pd.Series of returns (daily simple or log), indexed by date.
        variant: "GARCH", "EGARCH", or "GJR".
        dist:    innovation distribution ("normal", "t", "skewt").
        p, q:    GARCH orders.
        o:       asymmetry term (if None and variant="GJR", defaults to 1).
        scale_100: scale returns by 100 for fitting (arch convention). Result is scaled back.

    Returns:
        pd.Series named 'garch_vol' aligned to `returns.index`, same (daily) scale as input.
    """
    r = returns.dropna().copy()
    if r.empty:
        return pd.Series([], dtype=float, name="garch_vol")

    r_fit = r * 100.0 if scale_100 else r

    if variant == "EGARCH":
        am = arch_model(r_fit, mean="Zero", vol="EGARCH", p=p, q=q, dist=dist)
    elif variant == "GJR":
        o = 1 if o is None else o
        am = arch_model(r_fit, mean="Zero", vol="GARCH", p=p, o=o, q=q, dist=dist)
    else:
        am = arch_model(r_fit, mean="Zero", vol="GARCH", p=p, q=q, dist=dist)

    res = am.fit(disp="off")

    vol = res.conditional_volatility.astype(float)
    if scale_100:
        vol = vol / 100.0  # back to the same scale as `returns`

    vol.name = "garch_vol"
    # align to original index (fill gaps with NaN to keep shape identical to input)
    if not vol.index.equals(returns.index):
        vol = vol.reindex(returns.index)

    return vol

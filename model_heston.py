# models/heston.py

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class HestonParams:
    """
    Parameters of the Heston stochastic volatility model.

    kappa: mean-reversion speed of variance
    theta: long-run variance level
    sigma: vol-of-vol (variance shock scale)
    rho:   correlation between price and variance Brownian motions
    v0:    initial variance
    """
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float


def simulate_heston_paths(
    n_steps: int,
    n_sims: int,
    dt: float,
    params: HestonParams,
    s0: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate price and variance paths using the Heston model (Euler–Maruyama).

    S_t+1 = S_t + S_t * sqrt(v_t * dt) * W1_t
    v_t+1 = v_t + kappa*(theta - v_t)*dt + sigma*sqrt(v_t*dt)*W2_t
    with corr(W1_t, W2_t) = rho

    Returns:
        prices  : (n_steps+1, n_sims) DataFrame of simulated prices
        variances: (n_steps+1, n_sims) DataFrame of simulated variances
    """
    rng = np.random.default_rng(seed)

    prices = np.zeros((n_steps + 1, n_sims), dtype=float)
    variances = np.zeros_like(prices)
    prices[0, :] = s0
    variances[0, :] = max(params.v0, 1e-12)

    rho = params.rho
    for t in range(1, n_steps + 1):
        z1 = rng.standard_normal(n_sims)
        z2 = rng.standard_normal(n_sims)
        w1 = z1
        w2 = rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * z2

        v_prev = np.maximum(variances[t - 1, :], 1e-12)
        dv = params.kappa * (params.theta - v_prev) * dt + params.sigma * np.sqrt(v_prev * dt) * w2
        v_new = np.maximum(v_prev + dv, 1e-12)
        variances[t, :] = v_new

        dS = prices[t - 1, :] * np.sqrt(v_prev * dt) * w1
        prices[t, :] = prices[t - 1, :] + dS

    return pd.DataFrame(prices), pd.DataFrame(variances)


def estimate_heston_vol(
    returns: pd.Series,
    window: int = 7,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    TEMPORARY PROXY for Heston-implied volatility until calibration is added.
    Uses rolling standard deviation of returns as a simple volatility estimate.

    Args:
        returns: series of asset returns (e.g., daily log or simple returns).
        window: rolling window length (in periods).
        min_periods: minimum observations to compute the std (defaults to window).

    Returns:
        pd.Series named 'heston_vol' with the rolling volatility.
    """
    if min_periods is None:
        min_periods = window
    vol = returns.rolling(window=window, min_periods=min_periods).std()
    vol.name = "heston_vol"
    return vol



def calibrate_heston_params_placeholder(*args, **kwargs) -> HestonParams:
    """
    Placeholder for a real calibration routine (to be implemented later).
    Returns a reasonable default set of parameters so the pipeline runs.
    """
    # Typical ballpark values (to be replaced by proper calibration):
    return HestonParams(kappa=2.0, theta=0.02, sigma=0.5, rho=-0.6, v0=0.02)

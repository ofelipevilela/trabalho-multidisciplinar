# models/heston.py

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Dict


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
    S0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    T: float,
    M: int,
    N: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate price and variance paths using the Heston model (Euler–Maruyama).
    
    This is the core simulation function that generates multiple Monte Carlo paths
    to forecast future volatility.

    Args:
        S0: initial price (typically 1.0 for normalized simulations)
        v0: initial variance
        r: risk-free rate (annualized)
        kappa: mean-reversion speed of variance
        theta: long-run variance level
        sigma: vol-of-vol (variance shock scale)
        rho: correlation between price and variance Brownian motions
        T: time horizon (in years, e.g., 30/252 for 30 days)
        M: number of time steps
        N: number of Monte Carlo simulations
        seed: random seed for reproducibility

    Returns:
        prices: (M+1, N) array of simulated prices
        variances: (M+1, N) array of simulated variances
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / M
    prices = np.zeros((M + 1, N))
    variances = np.zeros((M + 1, N))
    prices[0, :] = S0
    variances[0, :] = max(v0, 1e-12)

    Z1 = np.random.normal(size=(M, N))
    Z2 = np.random.normal(size=(M, N))
    dW1 = np.sqrt(dt) * Z1
    dW2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

    epsilon = 1e-10
    
    for t in range(1, M + 1):
        # Variance process: mean-reverting with stochastic shocks
        v_prev = np.maximum(variances[t - 1, :], 1e-12)
        variances[t] = (
            v_prev 
            + kappa * (theta - v_prev) * dt 
            + sigma * np.sqrt(v_prev + epsilon) * dW2[t - 1]
        )
        variances[t] = np.maximum(variances[t], 1e-12)
        
        # Price process: geometric Brownian motion with stochastic volatility
        prices[t] = prices[t - 1] * np.exp(
            (r - 0.5 * v_prev) * dt 
            + np.sqrt(v_prev + epsilon) * dW1[t - 1]
        )
    
    return prices, variances


def _get_default_heston_params(ticker: str = "^GSPC") -> Dict[str, float]:
    """
    Returns default Heston parameters for S&P 500.
    
    These are standard parameters that can be calibrated later.
    Currently using S&P 500 defaults for all assets to ensure consistency.
    
    Args:
        ticker: asset ticker symbol (currently using S&P 500 params for all)
        
    Returns:
        Dictionary with kappa, theta, sigma, rho, and r
    """
    # S&P 500 default parameters (to be calibrated)
    return {
        "kappa": 2.5,   # mean-reversion speed
        "theta": 0.025, # long-run variance (vol ~15.8% annual)
        "sigma": 0.55,  # vol-of-vol
        "rho": -0.5,    # correlation (leverage effect)
        "r": 0.02,      # risk-free rate (2% annual)
    }


def estimate_heston_vol(
    returns: pd.Series,
    window: int = 7,
    min_periods: Optional[int] = None,
    ticker: str = "^GSPC",
    n_sims: int = 500,  # Reduced for faster execution during testing
    forecast_days: int = 30,
    n_steps: int = 30,
    v0_window: int = 22,
    verbose: bool = False,
) -> pd.Series:
    """
    Estimate Heston-implied volatility using Monte Carlo simulation.
    
    For each day, this function:
    1. Calculates initial variance (v0) from recent historical returns
    2. Runs Monte Carlo simulations using the Heston model
    3. Forecasts future volatility and returns the mean forecast
    
    Args:
        returns: series of asset returns (daily simple or log returns)
        window: DEPRECATED - kept for compatibility, not used
        min_periods: DEPRECATED - kept for compatibility, not used
        ticker: asset ticker symbol (for parameter selection)
        n_sims: number of Monte Carlo simulations per day
        forecast_days: forecast horizon in days (default 30)
        n_steps: number of time steps in simulation (default 30)
        v0_window: window size for calculating initial variance (default 22 days)
    
    Returns:
        pd.Series named 'heston_vol' with daily volatility forecasts (same scale as input returns)
    """
    r = returns.dropna().copy()
    if r.empty:
        return pd.Series([], dtype=float, name="heston_vol")
    
    # Get Heston parameters
    params = _get_default_heston_params(ticker)
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    rho = params["rho"]
    r_rate = params["r"]
    
    # Simulation parameters
    T = forecast_days / 252.0  # Convert days to years
    M = n_steps
    N = n_sims
    S0 = 1.0  # Normalized initial price
    
    # Initialize output series
    vol_forecasts = []
    
    if verbose:
        print(f"\n[Heston] Starting volatility estimation for {ticker}")
        print(f"  Parameters: kappa={kappa:.2f}, theta={theta:.4f}, sigma={sigma:.2f}, rho={rho:.2f}")
        print(f"  Simulation: {n_sims} paths, {forecast_days} days ahead, {n_steps} steps")
        print(f"  Total days to process: {len(r)}")
    
    for i in range(len(r)):
        # Calculate initial variance (v0) from historical returns
        if i < v0_window:
            # Use long-run variance if not enough history
            v0 = float(theta)
        else:
            # Use realized variance from last v0_window days
            historical_returns = r.iloc[i - v0_window:i]
            if len(historical_returns.dropna()) >= v0_window:
                # Annualized variance: (daily_std * sqrt(252))^2
                # Ensure we get a float value, not a Series
                daily_std = float(historical_returns.std())
                v0 = float((daily_std * np.sqrt(252)) ** 2)
            else:
                v0 = float(theta)
        
        # Ensure v0 is a float
        v0 = float(v0)
        
        # Run Monte Carlo simulation
        try:
            _, simulated_variances = simulate_heston_paths(
                S0, v0, r_rate, kappa, theta, sigma, rho, T, M, N
            )
            
            # Get mean future variance at the end of forecast horizon
            mean_future_variance = float(np.mean(simulated_variances[-1]))
            
            # Convert to daily volatility (not annualized)
            # If variance is annualized, convert: sqrt(variance) / sqrt(252)
            # If variance is already daily, just take sqrt
            # We assume the simulation produces annualized variance, so convert back
            daily_vol = float(np.sqrt(mean_future_variance) / np.sqrt(252))
            
        except Exception as e:
            # Fallback to simple rolling std if simulation fails
            if i >= v0_window:
                daily_vol = float(r.iloc[max(0, i - v0_window):i].std())
            else:
                daily_vol = float(np.sqrt(theta) / np.sqrt(252))
        
        # Ensure daily_vol is a simple float, not a Series
        daily_vol = float(daily_vol) if not isinstance(daily_vol, (int, float)) else daily_vol
        vol_forecasts.append(float(daily_vol))
        
        # Progress indicator for long runs
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(r)} days...")
    
    # Create output series aligned with input index
    # Ensure all values are floats and index is clean
    vol_forecasts_array = np.array(vol_forecasts, dtype=float)
    
    # Ensure index is a simple DatetimeIndex (not MultiIndex)
    clean_index = r.index
    if isinstance(clean_index, pd.MultiIndex):
        # If MultiIndex, use the first level (usually dates)
        clean_index = clean_index.get_level_values(0)
    
    vol = pd.Series(vol_forecasts_array, index=clean_index, name="heston_vol", dtype=float)
    
    if verbose:
        mean_vol = float(vol.mean())
        std_vol = float(vol.std())
        print(f"  ✓ Completed! Mean vol: {mean_vol:.6f}, Std: {std_vol:.6f}")
    
    # Reindex to match original returns index (fill gaps with NaN)
    # Ensure returns.index is also clean
    clean_returns_index = returns.index
    if isinstance(clean_returns_index, pd.MultiIndex):
        clean_returns_index = clean_returns_index.get_level_values(0)
    
    if not vol.index.equals(clean_returns_index):
        vol = vol.reindex(clean_returns_index)
    
    return vol


def calibrate_heston_params_placeholder(*args, **kwargs) -> HestonParams:
    """
    Placeholder for a real calibration routine (to be implemented later).
    Returns a reasonable default set of parameters so the pipeline runs.
    """
    # Typical ballpark values (to be replaced by proper calibration):
    return HestonParams(kappa=2.0, theta=0.02, sigma=0.5, rho=-0.6, v0=0.02)

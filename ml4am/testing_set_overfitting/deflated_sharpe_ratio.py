"""
Testing Set Overfitting - Deflated Sharpe Ratio

This module implements the concepts from Chapter 8 of "Machine Learning for Asset Managers"
by Marcos Lopez de Prado, focusing on:
- The False Strategy Theorem
- Deflated Sharpe Ratio (DSR)
- Type I and Type II errors under multiple testing
- Selection Bias under Multiple Testing (SBuMT)

References:
- Bailey, D.H. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management, 40(5)
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import norm
from typing import Union, List, Optional


# Euler-Mascheroni constant
EMC = 0.5772156649015328606065120900824024310421593359


def expected_max_sharpe_ratio(n_trials: int,
                               mean_sr: float = 0,
                               std_sr: float = 1) -> float:
    """
    Compute the expected maximum Sharpe ratio from n_trials independent trials.

    This implements the False Strategy Theorem which shows that when running
    multiple backtests, the expected maximum SR increases with the number of trials,
    even when the true SR is zero.

    Parameters
    ----------
    n_trials : int
        Number of independent trials/backtests
    mean_sr : float, default=0
        Mean Sharpe ratio across trials (null hypothesis assumes 0)
    std_sr : float, default=1
        Standard deviation of Sharpe ratios across trials

    Returns
    -------
    float
        Expected maximum Sharpe ratio

    Notes
    -----
    Based on the approximation:
    E[max(SR)] ≈ (1-γ)Φ^(-1)(1-1/N) + γΦ^(-1)(1-(Ne)^(-1))
    where γ is the Euler-Mascheroni constant
    """
    # Handle edge case where n_trials = 1
    if n_trials <= 1:
        return mean_sr

    # Approximation for expected maximum of N standard normal variables
    sr0 = (1 - EMC) * norm.ppf(1 - 1. / n_trials) + \
          EMC * norm.ppf(1 - (n_trials * np.e) ** -1)
    # Scale by mean and std
    sr0 = mean_sr + std_sr * sr0
    return sr0


def simulate_max_sharpe_distribution(n_sims: int,
                                      n_trials_list: List[int],
                                      mean_sr: float = 0,
                                      std_sr: float = 1,
                                      random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Simulate the distribution of maximum Sharpe ratios across multiple trials.

    Parameters
    ----------
    n_sims : int
        Number of simulations to run
    n_trials_list : list of int
        List of different trial counts to simulate
    mean_sr : float, default=0
        Mean Sharpe ratio
    std_sr : float, default=1
        Standard deviation of Sharpe ratios
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'max_sr' and 'n_trials'
    """
    rng = np.random.RandomState(random_state)
    results = []

    for n_trials in n_trials_list:
        # Handle edge case of n_trials = 1
        if n_trials <= 1:
            # For single trial, max = the single value itself
            max_sr = rng.randn(n_sims) * std_sr + mean_sr
        else:
            # Generate random SRs for each simulation
            sr = pd.DataFrame(rng.randn(n_sims, n_trials))
            # Standardize across trials
            sr = sr.sub(sr.mean(axis=1), axis=0)
            sr = sr.div(sr.std(axis=1), axis=0)
            # Scale to desired distribution
            sr = mean_sr + sr * std_sr
            # Get maximum SR for each simulation
            max_sr = sr.max(axis=1)

        df = pd.DataFrame({
            'max_sr': max_sr,
            'n_trials': n_trials
        })
        results.append(df)

    return pd.concat(results, ignore_index=True)


def get_z_statistic(sr: float,
                    n_obs: int,
                    sr_benchmark: float = 0,
                    skewness: float = 0,
                    kurtosis: float = 3) -> float:
    """
    Compute the z-statistic for a Sharpe ratio, adjusted for non-normality.

    Parameters
    ----------
    sr : float
        Observed Sharpe ratio
    n_obs : int
        Number of observations (sample length)
    sr_benchmark : float, default=0
        Benchmark Sharpe ratio (null hypothesis)
    skewness : float, default=0
        Skewness of returns
    kurtosis : float, default=3
        Kurtosis of returns (3 for normal distribution)

    Returns
    -------
    float
        Z-statistic

    Notes
    -----
    The standard error of the Sharpe ratio is adjusted for non-normality:
    SE(SR) = sqrt((1 - γ₁*SR + (γ₂-1)/4 * SR²) / (T-1))
    where γ₁ is skewness and γ₂ is kurtosis
    """
    z = (sr - sr_benchmark) * (n_obs - 1) ** 0.5
    z /= (1 - skewness * sr + (kurtosis - 1) / 4. * sr ** 2) ** 0.5
    return z


def type1_error_probability(z: float, n_trials: int = 1) -> float:
    """
    Compute Type I error probability (false positive rate) under multiple testing.

    Type I error is the probability of rejecting a true null hypothesis
    (finding a strategy when none exists).

    Parameters
    ----------
    z : float
        Z-statistic
    n_trials : int, default=1
        Number of independent trials (for Bonferroni-like correction)

    Returns
    -------
    float
        Familywise Type I error probability (alpha_k)

    Notes
    -----
    α_k = 1 - (1 - α)^k
    where α is the single-test error and k is the number of trials
    """
    alpha = ss.norm.cdf(-z)  # Single test p-value
    alpha_k = 1 - (1 - alpha) ** n_trials  # Familywise error rate
    return alpha_k


def get_theta(sr: float,
              n_obs: int,
              sr_benchmark: float = 0,
              skewness: float = 0,
              kurtosis: float = 3) -> float:
    """
    Compute the non-centrality parameter theta for Type II error calculation.

    Parameters
    ----------
    sr : float
        True Sharpe ratio
    n_obs : int
        Number of observations
    sr_benchmark : float, default=0
        Benchmark Sharpe ratio
    skewness : float, default=0
        Skewness of returns
    kurtosis : float, default=3
        Kurtosis of returns

    Returns
    -------
    float
        Non-centrality parameter theta
    """
    theta = sr_benchmark * (n_obs - 1) ** 0.5
    theta /= (1 - skewness * sr + (kurtosis - 1) / 4. * sr ** 2) ** 0.5
    return theta


def type2_error_probability(alpha_k: float,
                            n_trials: int,
                            theta: float) -> float:
    """
    Compute Type II error probability (false negative rate) under multiple testing.

    Type II error is the probability of failing to reject a false null hypothesis
    (missing a true strategy).

    Parameters
    ----------
    alpha_k : float
        Familywise Type I error probability
    n_trials : int
        Number of independent trials
    theta : float
        Non-centrality parameter

    Returns
    -------
    float
        Type II error probability (beta)
    """
    z = ss.norm.ppf((1 - alpha_k) ** (1. / n_trials))
    beta = ss.norm.cdf(z - theta)
    return beta


def deflated_sharpe_ratio(sr_observed: float,
                          sr_benchmark: float,
                          n_obs: int,
                          n_trials: int,
                          skewness: float = 0,
                          kurtosis: float = 3,
                          return_pvalue: bool = True) -> float:
    """
    Compute the Deflated Sharpe Ratio (DSR).

    The DSR is the probability that the observed Sharpe ratio is statistically
    significant after accounting for:
    1. Non-normality (skewness and kurtosis)
    2. Sample length
    3. Multiple testing (selection bias)

    Parameters
    ----------
    sr_observed : float
        Observed Sharpe ratio from backtest
    sr_benchmark : float
        Benchmark or expected maximum SR under null hypothesis
        (typically computed using expected_max_sharpe_ratio)
    n_obs : int
        Number of observations (backtest length)
    n_trials : int
        Number of trials/backtests conducted
    skewness : float, default=0
        Skewness of returns
    kurtosis : float, default=3
        Kurtosis of returns
    return_pvalue : bool, default=True
        If True, return p-value (DSR). If False, return z-statistic.

    Returns
    -------
    float
        DSR (probability) if return_pvalue=True, else z-statistic

    Notes
    -----
    A DSR close to 1 indicates the strategy is likely genuine.
    A DSR close to 0 indicates the strategy is likely the result of overfitting.

    The benchmark SR should account for multiple testing:
    sr_benchmark = expected_max_sharpe_ratio(n_trials, 0, std_sr)
    """
    # Compute z-statistic adjusted for non-normality
    z = get_z_statistic(sr_observed, n_obs, sr_benchmark, skewness, kurtosis)

    if return_pvalue:
        # DSR is the probability of observing SR >= sr_observed
        # given that true SR = sr_benchmark
        return ss.norm.cdf(z)
    else:
        return z


def probabilistic_sharpe_ratio(sr_observed: float,
                                sr_benchmark: float,
                                n_obs: int,
                                skewness: float = 0,
                                kurtosis: float = 3) -> float:
    """
    Compute the Probabilistic Sharpe Ratio (PSR).

    PSR is a simplified version of DSR that doesn't account for multiple testing,
    but does adjust for non-normality and sample length.

    Parameters
    ----------
    sr_observed : float
        Observed Sharpe ratio
    sr_benchmark : float
        Benchmark Sharpe ratio to beat
    n_obs : int
        Number of observations
    skewness : float, default=0
        Skewness of returns
    kurtosis : float, default=3
        Kurtosis of returns

    Returns
    -------
    float
        Probability that the true SR exceeds sr_benchmark
    """
    z = get_z_statistic(sr_observed, n_obs, sr_benchmark, skewness, kurtosis)
    return ss.norm.cdf(z)


def min_backtest_length(sr_target: float,
                        sr_benchmark: float = 0,
                        skewness: float = 0,
                        kurtosis: float = 3,
                        confidence: float = 0.95) -> int:
    """
    Compute the minimum backtest length required for statistical significance.

    Parameters
    ----------
    sr_target : float
        Target annualized Sharpe ratio
    sr_benchmark : float, default=0
        Benchmark Sharpe ratio
    skewness : float, default=0
        Skewness of returns
    kurtosis : float, default=3
        Kurtosis of returns
    confidence : float, default=0.95
        Required confidence level

    Returns
    -------
    int
        Minimum number of observations required
    """
    z_critical = ss.norm.ppf(confidence)
    sr_diff = sr_target - sr_benchmark

    # Solve for n: z = (SR - SR*) * sqrt(n-1) / SE
    # where SE = sqrt(1 - γ₁*SR + (γ₂-1)/4 * SR²)
    se_factor = 1 - skewness * sr_target + (kurtosis - 1) / 4. * sr_target ** 2

    n_min = 1 + (z_critical ** 2 * se_factor) / (sr_diff ** 2)
    return int(np.ceil(n_min))


def haircut_sharpe_ratio(sr_observed: float,
                         n_obs: int,
                         n_trials: int,
                         autocorrelation: float = 0) -> float:
    """
    Apply a haircut to the observed Sharpe ratio to account for overfitting.

    Parameters
    ----------
    sr_observed : float
        Observed Sharpe ratio
    n_obs : int
        Number of observations
    n_trials : int
        Number of trials conducted
    autocorrelation : float, default=0
        First-order autocorrelation of returns

    Returns
    -------
    float
        Haircut (adjusted) Sharpe ratio

    Notes
    -----
    The haircut accounts for:
    1. Multiple testing bias (more trials = larger haircut)
    2. Autocorrelation (positive autocorrelation inflates SR)
    """
    # Expected maximum SR under null
    sr_expected_max = expected_max_sharpe_ratio(n_trials, 0, 1)

    # Autocorrelation adjustment
    # SR_adjusted = SR * sqrt((1 - ρ) / (1 + ρ))
    if autocorrelation != 0:
        ac_adjustment = np.sqrt((1 - autocorrelation) / (1 + autocorrelation))
        sr_observed = sr_observed * ac_adjustment

    # Apply haircut
    sr_haircut = sr_observed - sr_expected_max / np.sqrt(n_obs)

    return max(0, sr_haircut)


class DeflatedSharpeRatio:
    """
    A class to compute and analyze Deflated Sharpe Ratios.

    This class provides a comprehensive interface for evaluating backtest
    results while accounting for multiple testing, non-normality, and
    selection bias.

    Parameters
    ----------
    n_trials : int
        Number of backtests/trials conducted
    mean_sr : float, default=0
        Mean Sharpe ratio under null hypothesis
    std_sr : float, default=1
        Standard deviation of Sharpe ratios under null

    Attributes
    ----------
    expected_max_sr : float
        Expected maximum SR under null hypothesis
    """

    def __init__(self,
                 n_trials: int,
                 mean_sr: float = 0,
                 std_sr: float = 1):
        self.n_trials = n_trials
        self.mean_sr = mean_sr
        self.std_sr = std_sr
        self.expected_max_sr = expected_max_sharpe_ratio(n_trials, mean_sr, std_sr)

    def evaluate(self,
                 sr_observed: float,
                 n_obs: int,
                 skewness: float = 0,
                 kurtosis: float = 3) -> dict:
        """
        Evaluate a backtest result.

        Parameters
        ----------
        sr_observed : float
            Observed Sharpe ratio
        n_obs : int
            Number of observations
        skewness : float, default=0
            Skewness of returns
        kurtosis : float, default=3
            Kurtosis of returns

        Returns
        -------
        dict
            Dictionary containing:
            - 'dsr': Deflated Sharpe Ratio
            - 'psr': Probabilistic Sharpe Ratio
            - 'expected_max_sr': Expected maximum SR under null
            - 'z_statistic': Z-statistic
            - 'is_significant': Whether DSR > 0.95
        """
        dsr = deflated_sharpe_ratio(
            sr_observed, self.expected_max_sr, n_obs, self.n_trials,
            skewness, kurtosis, return_pvalue=True
        )

        psr = probabilistic_sharpe_ratio(
            sr_observed, 0, n_obs, skewness, kurtosis
        )

        z = get_z_statistic(sr_observed, n_obs, self.expected_max_sr, skewness, kurtosis)

        return {
            'dsr': dsr,
            'psr': psr,
            'expected_max_sr': self.expected_max_sr,
            'z_statistic': z,
            'is_significant': dsr > 0.95
        }

    def __repr__(self):
        return (f"DeflatedSharpeRatio(n_trials={self.n_trials}, "
                f"expected_max_sr={self.expected_max_sr:.4f})")

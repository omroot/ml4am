"""
Probability of Backtest Overfitting (PBO)

This module implements the Combinatorially Symmetric Cross-Validation (CSCV) framework
for computing the Probability of Backtest Overfitting (PBO).

References:
- Bailey, D.H., Borwein, J., Lopez de Prado, M., and Zhu, Q.J. (2017),
  "The Probability of Backtest Overfitting", Journal of Computational Finance, 20(4)
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from typing import Tuple, List, Optional, Union
import warnings


def generate_partitions(n_groups: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Generate all combinatorially symmetric partitions of n_groups into two equal halves.

    For CSCV, we split the data into n_groups sub-samples, then enumerate all ways
    to partition these into training (S) and testing (S̄) sets of equal size.

    Parameters
    ----------
    n_groups : int
        Number of groups (must be even)

    Returns
    -------
    list of tuples
        Each tuple contains (train_indices, test_indices)

    Notes
    -----
    The number of partitions is C(n_groups, n_groups/2) / 2
    For n_groups=16, this gives 6,435 partitions
    """
    if n_groups % 2 != 0:
        raise ValueError("n_groups must be even for symmetric partitioning")

    half = n_groups // 2
    all_indices = list(range(n_groups))

    # Get all combinations of size n_groups/2 for training set
    train_combos = list(combinations(all_indices, half))

    # Filter to avoid duplicates (S, S̄) and (S̄, S) are the same partition
    partitions = []
    seen = set()

    for train_idx in train_combos:
        test_idx = tuple(i for i in all_indices if i not in train_idx)
        # Create canonical form (smaller first element goes first)
        if train_idx < test_idx:
            key = (train_idx, test_idx)
        else:
            key = (test_idx, train_idx)

        if key not in seen:
            seen.add(key)
            partitions.append((train_idx, test_idx))

    return partitions


def compute_performance_matrix(returns_matrix: np.ndarray,
                               performance_func: Optional[callable] = None) -> np.ndarray:
    """
    Compute performance for each strategy across each time group.

    Parameters
    ----------
    returns_matrix : np.ndarray
        Matrix of shape (n_observations, n_strategies) containing returns
    performance_func : callable, optional
        Function to compute performance metric. Default is Sharpe ratio.
        Should accept a 1D array of returns and return a scalar.

    Returns
    -------
    np.ndarray
        Performance matrix of shape (n_groups, n_strategies)
    """
    if performance_func is None:
        def performance_func(r):
            if len(r) == 0 or np.std(r) == 0:
                return 0.0
            return np.mean(r) / np.std(r) * np.sqrt(252)  # Annualized SR

    return performance_func(returns_matrix)


class CSCV:
    """
    Combinatorially Symmetric Cross-Validation for computing
    the Probability of Backtest Overfitting (PBO).

    This class implements the CSCV framework to assess whether a backtested
    strategy is likely to be overfit.

    Parameters
    ----------
    n_groups : int, default=16
        Number of groups to split the data into (must be even).
        More groups = more partitions = more robust estimate, but slower.

    Attributes
    ----------
    n_partitions : int
        Number of combinatorial partitions
    pbo : float
        Probability of Backtest Overfitting (after fit)
    logits : np.ndarray
        Logit values for each partition (after fit)
    performance_degradation : float
        Average performance degradation from in-sample to out-of-sample

    Examples
    --------
    >>> # Simulate returns for multiple strategies
    >>> np.random.seed(42)
    >>> n_obs, n_strategies = 1000, 100
    >>> returns = np.random.randn(n_obs, n_strategies) * 0.01
    >>> # Add some signal to a few strategies
    >>> returns[:, :5] += 0.0005
    >>>
    >>> cscv = CSCV(n_groups=16)
    >>> results = cscv.fit(returns)
    >>> print(f"PBO: {results['pbo']:.2%}")
    """

    def __init__(self, n_groups: int = 16):
        if n_groups % 2 != 0:
            raise ValueError("n_groups must be even")
        if n_groups < 4:
            raise ValueError("n_groups must be at least 4")

        self.n_groups = n_groups
        self.partitions = generate_partitions(n_groups)
        self.n_partitions = len(self.partitions)

        # Results (populated after fit)
        self.pbo = None
        self.logits = None
        self.performance_degradation = None
        self.lambda_values = None
        self.is_fit = False

    def fit(self,
            returns_matrix: np.ndarray,
            performance_func: Optional[callable] = None) -> dict:
        """
        Fit CSCV and compute PBO.

        Parameters
        ----------
        returns_matrix : np.ndarray
            Matrix of shape (n_observations, n_strategies) containing returns
            for each strategy over time.
        performance_func : callable, optional
            Function to compute performance from returns.
            Default is annualized Sharpe ratio.

        Returns
        -------
        dict
            Dictionary containing:
            - 'pbo': Probability of Backtest Overfitting
            - 'logits': Array of logit values for each partition
            - 'performance_degradation': Avg degradation from IS to OOS
            - 'lambda_values': Relative rank of best IS strategy in OOS
            - 'n_partitions': Number of partitions evaluated
        """
        if performance_func is None:
            def performance_func(r):
                if len(r) == 0 or np.std(r) == 0:
                    return 0.0
                return np.mean(r) / np.std(r) * np.sqrt(252)

        n_obs, n_strategies = returns_matrix.shape
        group_size = n_obs // self.n_groups

        # Split data into groups
        groups = []
        for i in range(self.n_groups):
            start = i * group_size
            end = start + group_size if i < self.n_groups - 1 else n_obs
            groups.append(returns_matrix[start:end, :])

        # Compute performance for each strategy in each group
        perf_by_group = np.zeros((self.n_groups, n_strategies))
        for i, group_returns in enumerate(groups):
            for j in range(n_strategies):
                perf_by_group[i, j] = performance_func(group_returns[:, j])

        # For each partition, find best IS strategy and compute OOS performance
        logits = []
        lambda_values = []
        is_performances = []
        oos_performances = []

        for train_idx, test_idx in self.partitions:
            # In-sample performance (average across training groups)
            is_perf = perf_by_group[list(train_idx), :].mean(axis=0)

            # Out-of-sample performance (average across test groups)
            oos_perf = perf_by_group[list(test_idx), :].mean(axis=0)

            # Find best strategy in-sample
            best_is_idx = np.argmax(is_perf)
            best_is_perf = is_perf[best_is_idx]
            best_is_oos_perf = oos_perf[best_is_idx]

            is_performances.append(best_is_perf)
            oos_performances.append(best_is_oos_perf)

            # Compute relative rank (lambda) of best IS strategy in OOS
            # lambda = rank / n_strategies, where rank 1 = best
            oos_rank = (oos_perf > best_is_oos_perf).sum() + 1
            lambda_val = oos_rank / n_strategies
            lambda_values.append(lambda_val)

            # Compute logit: log(lambda / (1 - lambda))
            # Clamp to avoid log(0) or log(inf)
            lambda_clamped = np.clip(lambda_val, 1e-10, 1 - 1e-10)
            logit = np.log(lambda_clamped / (1 - lambda_clamped))
            logits.append(logit)

        self.logits = np.array(logits)
        self.lambda_values = np.array(lambda_values)

        # PBO = proportion of partitions where logit > 0 (i.e., lambda > 0.5)
        # which means the best IS strategy ranks below median OOS
        self.pbo = (self.logits > 0).mean()

        # Performance degradation
        is_performances = np.array(is_performances)
        oos_performances = np.array(oos_performances)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.performance_degradation = np.nanmean(
                (is_performances - oos_performances) / np.abs(is_performances)
            )

        self.is_fit = True

        return {
            'pbo': self.pbo,
            'logits': self.logits,
            'lambda_values': self.lambda_values,
            'performance_degradation': self.performance_degradation,
            'n_partitions': self.n_partitions,
            'is_performance_mean': np.mean(is_performances),
            'oos_performance_mean': np.mean(oos_performances)
        }

    def get_stochastic_dominance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the stochastic dominance curve for visualization.

        Returns
        -------
        tuple of (x, y)
            x: sorted lambda values
            y: cumulative probability
        """
        if not self.is_fit:
            raise ValueError("Must call fit() first")

        sorted_lambda = np.sort(self.lambda_values)
        cumulative_prob = np.arange(1, len(sorted_lambda) + 1) / len(sorted_lambda)

        return sorted_lambda, cumulative_prob

    def __repr__(self):
        status = f"PBO={self.pbo:.2%}" if self.is_fit else "not fit"
        return f"CSCV(n_groups={self.n_groups}, n_partitions={self.n_partitions}, {status})"


def probability_of_overfitting(returns_matrix: np.ndarray,
                               n_groups: int = 16,
                               performance_func: Optional[callable] = None) -> dict:
    """
    Convenience function to compute PBO using CSCV.

    Parameters
    ----------
    returns_matrix : np.ndarray
        Matrix of shape (n_observations, n_strategies)
    n_groups : int, default=16
        Number of groups for CSCV
    performance_func : callable, optional
        Performance metric function

    Returns
    -------
    dict
        Results including PBO, logits, and other statistics

    Examples
    --------
    >>> returns = np.random.randn(1000, 50) * 0.01
    >>> result = probability_of_overfitting(returns)
    >>> print(f"PBO: {result['pbo']:.2%}")
    """
    cscv = CSCV(n_groups=n_groups)
    return cscv.fit(returns_matrix, performance_func)


def minimum_backtest_length_pbo(n_trials: int,
                                 target_sr: float = 1.0,
                                 max_pbo: float = 0.05) -> int:
    """
    Estimate minimum backtest length to achieve target PBO.

    This is based on the theoretical relationship between number of trials,
    backtest length, and probability of overfitting.

    Parameters
    ----------
    n_trials : int
        Number of strategy configurations tested
    target_sr : float, default=1.0
        Target Sharpe ratio
    max_pbo : float, default=0.05
        Maximum acceptable PBO (e.g., 0.05 for 5%)

    Returns
    -------
    int
        Minimum number of observations required

    Notes
    -----
    This is an approximation based on the relationship:
    MinBTL ≈ (z_α + z_β)² / SR² * (1 + log(N)/2)

    where N is the number of trials.
    """
    z_alpha = stats.norm.ppf(1 - max_pbo)
    z_beta = stats.norm.ppf(0.8)  # 80% power

    # Adjustment for multiple testing
    multiple_testing_factor = 1 + np.log(n_trials) / 2

    min_length = ((z_alpha + z_beta) ** 2 / target_sr ** 2) * multiple_testing_factor

    return int(np.ceil(min_length * 252))  # Convert to daily observations

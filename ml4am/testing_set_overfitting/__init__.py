"""
Testing Set Overfitting Module

This module implements Chapter 8 of "Machine Learning for Asset Managers"
by Marcos Lopez de Prado.

Key concepts:
- False Strategy Theorem
- Deflated Sharpe Ratio (DSR)
- Probabilistic Sharpe Ratio (PSR)
- Type I and Type II errors under multiple testing
- Selection Bias under Multiple Testing (SBuMT)
- Probability of Backtest Overfitting (PBO)
- Combinatorially Symmetric Cross-Validation (CSCV)
"""

from .deflated_sharpe_ratio import (
    # Core functions
    expected_max_sharpe_ratio,
    simulate_max_sharpe_distribution,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,

    # Error analysis
    get_z_statistic,
    type1_error_probability,
    type2_error_probability,
    get_theta,

    # Utility functions
    min_backtest_length,
    haircut_sharpe_ratio,

    # Class interface
    DeflatedSharpeRatio,

    # Constants
    EMC,
)

from .probability_of_overfitting import (
    # CSCV / PBO
    CSCV,
    probability_of_overfitting,
    generate_partitions,
    minimum_backtest_length_pbo,
)

__all__ = [
    # Deflated Sharpe Ratio
    'expected_max_sharpe_ratio',
    'simulate_max_sharpe_distribution',
    'deflated_sharpe_ratio',
    'probabilistic_sharpe_ratio',
    'get_z_statistic',
    'type1_error_probability',
    'type2_error_probability',
    'get_theta',
    'min_backtest_length',
    'haircut_sharpe_ratio',
    'DeflatedSharpeRatio',
    'EMC',
    # Probability of Overfitting
    'CSCV',
    'probability_of_overfitting',
    'generate_partitions',
    'minimum_backtest_length_pbo',
]

# ML4AM - Machine Learning for Asset Managers

**Author:** Oualid Missaoui

A Python implementation of the techniques from **"Machine Learning for Asset Managers"** by Marcos López de Prado (Cambridge University Press, 2020).

## Overview

This library provides tools for applying machine learning to quantitative finance and portfolio management, including:

- **Denoising & Detoning** correlation matrices using Random Matrix Theory
- **Distance Metrics** based on information theory (mutual information, variation of information)
- **Optimal Clustering** with automatic cluster number selection
- **Feature Importance** analysis (MDI, MDA, clustered importance)
- **Financial Labels** generation via trend scanning
- **Portfolio Construction** using Nested Clustering Optimization (NCO)
- **Backtest Overfitting** detection (Deflated Sharpe Ratio, PBO)

## Installation

```bash
git clone https://github.com/yourusername/ML4AM.git
cd ML4AM
pip install -r requirements.txt
```

### Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- statsmodels
- joblib

## Project Structure

```
ML4AM/
├── ml4am/                          # Main library
│   ├── denoise_detone/             # Ch.2: Denoising & Detoning
│   │   ├── CleanseMatrix.py        # Main denoising class
│   │   ├── MarcenkoPastur.py       # Marcenko-Pastur distribution fitting
│   │   └── KernelDensityEstimator.py
│   │
│   ├── distance_metrics/           # Ch.3: Distance Metrics
│   │   └── distance_metrics.py     # Mutual info, variation of info
│   │
│   ├── optimal_clustering/         # Ch.4: Optimal Clustering
│   │   └── OptimalClustering.py    # ONC algorithm with silhouette/gap statistic
│   │
│   ├── financial_labels/           # Ch.5: Financial Labels
│   │   └── trend_scanning.py       # Trend scanning for label generation
│   │
│   ├── feature_importance/         # Ch.6: Feature Importance
│   │   └── feature_importance.py   # MDI, MDA, clustered importance
│   │
│   ├── portfolio_construction/     # Ch.7: Portfolio Construction
│   │   └── nco.py                  # Nested Clustering Optimization
│   │
│   ├── testing_set_overfitting/    # Ch.8: Testing Set Overfitting
│   │   ├── deflated_sharpe_ratio.py    # DSR, PSR, False Strategy Theorem
│   │   └── probability_of_overfitting.py # PBO with CSCV
│   │
│   └── datasets/                   # Synthetic data generators
│       └── mean_covariance_generator.py
│
├── notebooks/                      # Jupyter notebooks (tutorials)
│   ├── 1_introduction.ipynb
│   ├── 2_denoising_refactored.ipynb
│   ├── 3_distance_metrics.ipynb
│   ├── 4_optimal_clustering.ipynb
│   ├── 5_financial_labels.ipynb
│   ├── 6_feature_importance.ipynb
│   ├── 7_portfolio_construction.ipynb
│   └── 8_testing_set_overfitting.ipynb
│
└── docs/                           # Documentation & references
    ├── machine-learning-for-asset-managers.pdf
    └── lectures/                   # Presentation slides
```

## Quick Start

### 1. Denoising a Correlation Matrix

```python
from ml4am.denoise_detone import CleanseMatrix
import numpy as np

# Generate a noisy correlation matrix
returns = np.random.randn(1000, 100)
corr_matrix = np.corrcoef(returns, rowvar=False)

# Denoise using Marcenko-Pastur
cleanser = CleanseMatrix(method='denoise')
corr_denoised = cleanser.fit_transform(corr_matrix, q=10)
```

### 2. Optimal Clustering

```python
from ml4am.optimal_clustering import OptimalClustering

# Find optimal number of clusters
onc = OptimalClustering(max_number_clusters=20, method='silhouette')
onc.fit(corr_matrix)

print(f"Optimal clusters: {onc.n_clusters}")
print(f"Cluster labels: {onc.labels_}")
```

### 3. Detecting Backtest Overfitting

```python
from ml4am.testing_set_overfitting import (
    DeflatedSharpeRatio,
    probability_of_overfitting
)

# Deflated Sharpe Ratio
dsr = DeflatedSharpeRatio(n_trials=100)
result = dsr.evaluate(sr_observed=1.5, n_obs=756, skewness=-0.5, kurtosis=5)
print(f"DSR: {result['dsr']:.2%}")
print(f"Significant: {result['is_significant']}")

# Probability of Backtest Overfitting (PBO)
# returns_matrix: shape (n_observations, n_strategies)
pbo_result = probability_of_overfitting(returns_matrix, n_groups=16)
print(f"PBO: {pbo_result['pbo']:.2%}")
```

### 4. Portfolio Construction with NCO

```python
from ml4am.portfolio_construction import NCO

# Nested Clustering Optimization
nco = NCO()
weights = nco.fit(covariance_matrix, expected_returns)
print(f"Portfolio weights: {weights}")
```

## Notebooks

The `notebooks/` directory contains detailed tutorials for each chapter:

| Notebook | Topic | Description |
|----------|-------|-------------|
| 1 | Introduction | Data generation and setup |
| 2 | Denoising | Marcenko-Pastur theorem, eigenvalue cleaning |
| 3 | Distance Metrics | Mutual information, variation of information |
| 4 | Optimal Clustering | Silhouette method, gap statistic, ONC algorithm |
| 5 | Financial Labels | Trend scanning for supervised learning |
| 6 | Feature Importance | MDI, MDA, handling substitution effects |
| 7 | Portfolio Construction | NCO algorithm for robust allocation |
| 8 | Testing Set Overfitting | DSR, PBO, CSCV framework |

## Lecture Slides

Presentation decks are available in `docs/lectures/` for teaching and reference:

| Chapter | Topic | File |
|---------|-------|------|
| 2 | Denoising & Detoning | [ML4AM_Ch2_DenoisingDetoning.pptx](docs/lectures/ML4AM_Ch2_DenoisingDetoning.pptx) |
| 4 | Optimal Clustering | [ML4AM_Ch4_OptimalClustering.pptx](docs/lectures/ML4AM_Ch4_OptimalClustering.pptx) |
| 6 | Feature Importance | [ML4AM_Ch6_FeatureImportance.pptx](docs/lectures/ML4AM_Ch6_FeatureImportance.pptx) |

These slides provide visual explanations of the key concepts and can be used for presentations or self-study.

## Key Concepts

### Denoising (Chapter 2)

Uses Random Matrix Theory to separate signal from noise in correlation matrices:
- Fits the Marcenko-Pastur distribution to identify noise eigenvalues
- Shrinks noise eigenvalues while preserving signal
- Optional detoning to remove market-wide effects

### Optimal Clustering (Chapter 4)

Determines the optimal number of clusters using:
- **Silhouette Method**: Maximizes clustering quality q = E[S]/sqrt(Var[S])
- **Gap Statistic**: Compares within-cluster dispersion to null reference

### Testing Set Overfitting (Chapter 8)

Two complementary approaches:

1. **Deflated Sharpe Ratio (DSR)**: Adjusts for multiple testing, non-normality, and sample length
2. **Probability of Backtest Overfitting (PBO)**: Uses CSCV to directly estimate overfitting probability

| PBO Value | Interpretation |
|-----------|----------------|
| < 5% | Strong evidence of genuine strategy |
| 5-25% | Acceptable, use caution |
| 25-50% | Concerning, likely some overfitting |
| > 50% | High probability of overfitting |

## References

- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
- Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio", *Journal of Portfolio Management*, 40(5).
- Bailey, D.H., Borwein, J., López de Prado, M., & Zhu, Q.J. (2017). "The Probability of Backtest Overfitting", *Journal of Computational Finance*, 20(4).

## License

This project is for educational purposes. Please refer to the original book for commercial applications.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

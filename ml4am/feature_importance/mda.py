from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from typing import Union, Optional, List, Dict, Any, Tuple

class FeatureImportanceCalculator:
    def __init__(self, clf: ClassifierMixin, n_splits: int = 10):
        """Initialize the Feature Importance Calculator.

        Args:
            clf (ClassifierMixin): The classifier to use for feature importance calculation.
            n_splits (int): Number of splits for cross-validation.
        """
        self.clf = clf
        self.n_splits = n_splits

    def _feat_imp_mda(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """Compute feature importance based on out-of-sample score reduction.

        Args:
            X (pd.DataFrame): Input features.
            y (Union[pd.Series, np.ndarray]): Target labels.

        Returns:
            pd.DataFrame: Mean feature importance statistics.
        """
        cv_gen = KFold(n_splits=self.n_splits)
        scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)

        for i, (train, test) in enumerate(cv_gen.split(X=X)):
            X0, y0 = X.iloc[train, :], y[train]
            X1, y1 = X.iloc[test, :], y[test]
            fit = self.clf.fit(X=X0, y=y0)

            # Prediction before shuffling
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, labels=fit.classes_)

            for j in X.columns:
                X1_ = X1.copy(deep=True)
                np.random.shuffle(X1_[j].values)  # Shuffle one column
                prob = fit.predict_proba(X1_)  # Prediction after shuffling
                scr1.loc[i, j] = -log_loss(y1, prob, labels=fit.classes_)

        imp = (-1 * scr1).add(scr0, axis=0)
        imp = imp / (-1 * scr1)
        imp_stats = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -0.5}, axis=1)
        return imp_stats

    def _feat_imp_mda_clustered(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], clusters: Dict[int, List[str]]
    ) -> pd.DataFrame:
        """Compute clustered feature importance based on out-of-sample score reduction.

        Args:
            X (pd.DataFrame): Input features.
            y (Union[pd.Series, np.ndarray]): Target labels.
            clusters (Dict[int, List[str]]): Cluster information.

        Returns:
            pd.DataFrame: Mean clustered feature importance statistics.
        """
        cv_gen = KFold(n_splits=self.n_splits)
        scr0, scr1 = pd.Series(), pd.DataFrame(columns=clusters.keys())

        for i, (train, test) in enumerate(cv_gen.split(X=X)):
            X0, y0 = X.iloc[train, :], y[train]
            X1, y1 = X.iloc[test, :], y[test]
            fit = self.clf.fit(X=X0, y=y0)
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, labels=fit.classes_)

            for cluster_name, cluster_indices in clusters.items():
                X1_ = X1.copy(deep=True)
                for k in cluster_indices:
                    np.random.shuffle(X1_[k].values)
                prob = fit.predict_proba(X1_)
                scr1.loc[i, cluster_name] = -log_loss(y1, prob, labels=fit.classes_)

        imp = (-1 * scr1).add(scr0, axis=0)
        imp = imp / (-1 * scr1)
        imp_stats = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -0.5}, axis=1)
        imp_stats.index = ['C_' + str(i) for i in imp_stats.index]
        return imp_stats

    def compute_feature_importance(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], clusters: Optional[Dict[int, List[str]]] = None
    ) -> pd.DataFrame:
        """Compute feature importance.

        Args:
            X (pd.DataFrame): Input features.
            y (Union[pd.Series, np.ndarray]): Target labels.
            clusters (Optional[Dict[int, List[str]]]): Cluster information if clustered feature importance is needed.

        Returns:
            pd.DataFrame: Mean feature importance statistics.
        """
        if clusters is None:
            return self._feat_imp_mda(X, y)
        else:
            return self._feat_imp_mda_clustered(X, y, clusters)

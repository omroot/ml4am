import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

class MDI:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = X.columns
        self.feature_importance = None
        self.mean_mdi = None
        self.std_mdi = None
        self.p_values = None

    def fit(self):
        self.model.fit(self.X, self.y)
        self.feature_importance = self.model.feature_importances_
        self.mean_mdi = np.mean(self.feature_importance)
        self.std_mdi = np.std(self.feature_importance)
        _, self.p_values = ttest_1samp(self.feature_importance, 0)

    def get_results(self):
        results = pd.DataFrame({
            'Feature': self.feature_names,
            'MDI': self.feature_importance,
            'P-value': self.p_values
        })
        return results

# Assuming X and y are your feature matrix and target vector
# Initialize and fit the LightGBM model
lgb_model = lgb.LGBMClassifier()
mdi_calculator = MDI(model=lgb_model, X=X, y=y)
mdi_calculator.fit()

# Get the results including feature importance, p-values, mean MDI, and std MDI
mdi_results = mdi_calculator.get_results()
mean_mdi = mdi_calculator.mean_mdi
std_mdi = mdi_calculator.std_mdi



import random
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from statsmodels.discrete.discrete_model import Logit

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import pandas as pd
import numpy as np



def generate_classification_data(n_features: int =100,
                                 n_informative: int=25,
                                 n_redundant: int=25,
                                 n_samples: int=10000,
                                 random_state: int=0,
                                 sigma_std: float=.0):
    # Generate a random dataset for a classification problem
    np.random.seed(random_state)
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features - n_redundant,
                               n_informative=n_informative,
                               n_redundant=0,
                               shuffle=False,
                               random_state=random_state)

    # cols = ['I_' + str(i) for i in range(n_informative)]
    # cols += ['N_' + str(i) for i in range(n_features - n_informative - n_redundant)]
    #
    cols = [f'I_{i}' for i in range(n_informative)]
    cols += [f'N_{i}' for i in range(n_features - n_informative - n_redundant)]

    X, y = pd.DataFrame(X, columns=cols), pd.Series(y)

    informative_indices = np.random.choice(range(n_informative), size=n_redundant)
    for k, j in enumerate(informative_indices):
        X['R_' + str(k)] = X['I_' + str(j)] + np.random.normal(size=X.shape[0]) * sigma_std

    return X, y



def feat_importance_mdi(fit, feat_names):
    # Feature importance based on IS mean impurity reduction
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = feat_names
    df0 = df0.replace(0, np.nan)  # Replace 0 with NaN because max_features=1
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std() * df0.shape[0] ** -0.5}, axis=1)
    imp /= imp['mean'].sum()
    return imp




def feat_imp_mda(clf, X, y, n_splits=10):
    # Feature importance based on out-of-sample score reduction
    cv_gen = KFold(n_splits=n_splits)
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)

    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        X0, y0 = X.iloc[train, :], y.iloc[train]
        X1, y1 = X.iloc[test, :], y.iloc[test]
        fit = clf.fit(X=X0, y=y0)

        # Prediction before shuffling
        prob = fit.predict_proba(X1)
        scr0.loc[i] = -log_loss(y1, prob, labels=clf.classes_)

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # Shuffle one column
            prob = fit.predict_proba(X1_)  # Prediction after shuffling
            scr1.loc[i, j] = -log_loss(y1, prob, labels=clf.classes_)

    imp = (-1 * scr1).add(scr0, axis=0)
    imp = imp / (-1 * scr1)
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -0.5}, axis=1)

    return imp


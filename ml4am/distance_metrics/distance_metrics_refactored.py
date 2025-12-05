import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score



def discretization_optimal_number_of_bins(number_of_observations: int,
                                          correlation: float=None)->int:
    """Identifies the optimal number of bins to discretize one (X) or two variables (X and Y).

    The method implements the following two papers:
    1- univariate case:
    Hacine-Gharbi, A., P. Ravier, R. Harba, and T. Mohamadi (2012): "Low Bias Histogram-Based Estimation of Mutual
    Information for Feature Selection." Pattern Recognition Letters, Vol. 33, pp. 1302–8.
    2-bi-variate case:
    Hacine-Gharbi, A., and P. Ravier (2018): "A Binning Formula of Bi-histogram for Joint Entropy Estimation
    Using Mean Square Error Minimization." Pattern Recognition Letters, Vol. 101, pp. 21–28.

    Args:
        number_of_observations: number of observation  of X (and Y)
        correlation: correlation between X and Y
    Returns:
        optimal_number_of_bins : optimal number of bins for discretization
    """

    if correlation is None:  # univariate case
        zeta = (8 + 324 * number_of_observations + 12 * (36 * number_of_observations + 729 * number_of_observations ** 2) ** .5) ** (1 / 3.)
        optimal_number_of_bins = round(zeta / 6. + 2. / (3 * zeta) + 1. / 3)
    else:  # bivariate case
        optimal_number_of_bins = round(2 ** -.5 * (1 + (1 + 24 * number_of_observations / (1. - correlation ** 2)) ** .5) ** .5)

    return int(optimal_number_of_bins)

def variation_of_information(X: np.ndarray,
                             Y: np.ndarray,
                             number_of_bins: int=None,
                             normalize: bool=False) -> float:
    """"Computes the variation of information between two variables X and Y
    Args:
        X:  an array with the observations of X
        Y:  an array with the observations of Y
        number_of_bins: number of bins to be used to discretize (use discretization_optimal_number_of_bins if None)
        normalize:  normalize the variation of information
    Returns:
        variation_of_info_XY: variation of information between X and Y

    """

    if number_of_bins is None:
        correlation_xy = np.corrcoef(X, Y)[0, 1]
        number_of_bins = discretization_optimal_number_of_bins(X.shape[0], corr=correlation_xy)

    cXY = np.histogram2d(X, Y, number_of_bins)[0]
    Entropy_X = ss.entropy(np.histogram(X, number_of_bins)[0])  # marginal
    Entropy_Y = ss.entropy(np.histogram(Y, number_of_bins)[0])  # marginal
    mutual_info_XY = mutual_info_score(None, None, contingency=cXY)
    variation_of_info_XY = Entropy_X + Entropy_Y - 2 * mutual_info_XY  # variation of information
    if normalize:
        JointEntropy_XY = Entropy_X + Entropy_Y - mutual_info_XY  # joint
        variation_of_info_XY = variation_of_info_XY / JointEntropy_XY  # normalized varaiation of information - Kraskov (2008)

    return variation_of_info_XY

def mutual_information(X,
                       Y,
                       number_of_bins=None,
                       normalize=False):
    """
    Args:
        X: an array with the observations of X
        Y: an array with the observations of Y
        number_of_bins: number of bins to be used to discretize (use discretization_optimal_number_of_bins if None)
        normalize: normalize the variation of information
    Returns:
        mutual_info_XY: mutual information between X and Y

    """

    if number_of_bins is None:
        correlation_xy = np.corrcoef(X, Y)[0, 1]
        number_of_bins = discretization_optimal_number_of_bins(X.shape[0], corr=correlation_xy)

    cXY = np.histogram2d(X,Y, number_of_bins)[0]
    mutual_info_XY = mutual_info_score(None, None, contingency=cXY)
    if normalize:
        Entropy_X = ss.entropy(np.histogram(X, number_of_bins)[0]) #marginal
        Entropy_Y = ss.entropy(np.histogram(Y, number_of_bins)[0]) #marginal
        mutual_info_XY /= min(Entropy_X, Entropy_Y) #normalized mutual information

    return mutual_info_XY


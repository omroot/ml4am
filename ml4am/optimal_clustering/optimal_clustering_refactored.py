from itertools import groupby
from operator import itemgetter
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.utils import check_random_state
from scipy.linalg import block_diag
import matplotlib.pylab as plt
import matplotlib


from ml4am.denoising_refactored import *

def shuffle_matrix(input_matrix: pd.DataFrame)->pd.DataFrame:
    new_indicies = list(range(input_matrix.shape[0]))
    random.shuffle(new_indicies)
    shuffled_matrix = input_matrix.iloc[new_indicies]  # reorder rows
    shuffled_matrix = shuffled_matrix.iloc[:, new_indicies]  # reorder columns
    return shuffled_matrix

class ONC(KMeans):
    def __init__(self,
                 is_input_correlation = True,
                 max_number_clusters=None,
                 number_initializations=10,
                 **kwargs):
        """ ONC Base Clustering method.
        This kmeans based  algorithm doesn't require specifying the number of clusters. Rather, the number of
        clusters are learned based on the silouhette method
        Args:
            is_input_correlation: a flag to indicate if the input data is a correlation matrix,
            max_number_clusters: the maximum number of clusters,
            number_initializations: the number of initializations ,
        """
        self.is_input_correlation =  is_input_correlation
        self.max_number_clusters = max_number_clusters
        self.number_initializations = number_initializations
        self.quality = None
        self.silhouette = None
        self.reordered_X = None
        super().__init__(n_init=number_initializations, **kwargs)

    def fit(self, X, y=None, sample_weight=None):

        if self.is_input_correlation:
            X = ((1 - X.fillna(0)) / 2) ** 0.5

        if self.max_number_clusters is None:
            self.max_number_clusters = X.shape[1] - 1

        optimal_cluster_centers_ = None
        optimal_labels_ = None
        optimal_inertia_ = None
        optimal_n_iter_ = None
        optimal_silhouette = None
        optimal_quality = -1

        for init in range(self.number_initializations):
            for k in range(2, self.max_number_clusters + 1):
                self.n_clusters = k
                super().fit(X, y=y, sample_weight=sample_weight)
                silhouette = silhouette_samples(X, self.labels_)
                quality = silhouette.mean() / silhouette.std()
                if optimal_silhouette is None or quality > optimal_quality:
                    optimal_silhouette = silhouette
                    optimal_cluster_centers_ = self.cluster_centers_
                    optimal_labels_ = self.labels_.copy()
                    optimal_inertia_ = self.inertia_
                    optimal_n_iter_ = self.n_iter_
                    optimal_quality = quality
        self.cluster_centers_ = optimal_cluster_centers_
        self.labels_ = optimal_labels_
        self.inertia_ = optimal_inertia_
        self.n_iter_ = optimal_n_iter_
        self.quality = optimal_quality
        self.silhouette = optimal_silhouette
        self.n_clusters = len(np.unique(self.labels_))


        new_indicies = np.argsort(self.labels_)
        reordered_X = X.iloc[new_indicies]  # reorder rows
        reordered_X = reordered_X.iloc[:, new_indicies]  # reorder columns

        self.reordered_X = reordered_X

        return self
    @property
    def clusters(self):
        cluster = {}
        for i in self.labels_:
            cluster[i] = np.where(self.labels_ == i)[0]
        return cluster



class PrunedONC(ONC):



    def evaluate_clusters(self, labels, silhouette):
        clusters = groupby(
            sorted(zip(labels, silhouette)),
            itemgetter(0)
        )
        for key, data in clusters:
            silhouettes = tuple(data)
            if silhouettes is not None:
                mean = np.mean(silhouettes, axis=0)[1]
                vol = np.std(silhouettes, axis=0)[1]
                if vol != 0.:
                    yield key, mean / vol
                else:
                    yield key, 0.
            else:
                yield key, None


    def fit(self,X, y=None, sample_weight=None):
        super().fit(X=X, y=y, sample_weight=sample_weight)
        cluster_qualities = tuple(self.evaluate_clusters(self.labels_,
                                                 self.silhouette))

        _, clusters_average_quality = np.mean(cluster_qualities, axis=0)
        clusters_redo = tuple(
            key for key, quality in cluster_qualities
            if quality < clusters_average_quality
        )
        if len(clusters_redo)>1:
            reclustering_indicies, = np.where(np.isin(self.labels_, clusters_redo))
            reclustering_data = X.iloc[reclustering_indicies, reclustering_indicies]
            reclustering_onc = ONC(
                max_number_clusters=min(
                    self.max_number_clusters,
                    reclustering_data.shape[1] - 1),
                number_initializations=self.number_initializations,
                init=self.init,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                random_state=self.random_state,
                copy_x=self.copy_x,
                algorithm=self.algorithm

            )
            reclustering_onc.fit(reclustering_data)
            self.merge(reclustering_model=reclustering_onc,
                       X=X,
                       clusters_redo=clusters_redo,
                       clusters_average_quality=clusters_average_quality)

    def merge(self,
              reclustering_model,
              X,
              clusters_redo,
              clusters_average_quality):


        reclustering_indicies, = np.where(
            np.isin(self.labels_, clusters_redo)
        )
        pruned_labels = self.labels_.copy()
        pruned_labels[reclustering_indicies] = reclustering_model.labels_ + max(self.labels_)
        X = ((1 - X.fillna(0)) / 2.) ** .5
        pruned_silhouette = silhouette_samples(X, pruned_labels)
        pruned_quality = pruned_silhouette.mean() / pruned_silhouette.std()
        cluster_qualities = tuple(self.evaluate_clusters(pruned_labels, pruned_silhouette))
        _, pruned_clusters_average_quality = np.mean( cluster_qualities ,
                                             axis=0)

        if pruned_clusters_average_quality > clusters_average_quality:
            self.labels_ = pruned_labels
            self.quality = pruned_quality
            self.silhouette = pruned_silhouette

        return self




def generate_sub_covariance(number_observations: int,
                              number_factors: int,
                              sigma: float,
                              random_state=None) -> np.ndarray:
    """"Generate a sub correlation matrix from highly correlated factors
    Args:
        number_observations: number of observations of underlying factors
        number_factors: number of factors
        sigma: variance of added noise
        random_state: random state

    Returns:
        covariance_matrix: generated covariance matrix
    """
    rng = check_random_state(random_state)
    if number_factors == 1:
        return np.ones((1,1))
    data = rng.normal(size=(number_observations, 1)) #array of normal rv
    data = np.repeat(data, number_factors, axis=1) #matrix of columns repeating rv. Simulate time-series of at least 100 elements.
    data += rng.normal(loc=0, scale=sigma, size=data.shape) #add two rv X~Y~N(0,1), Z=X+Y~N(0+0, 1+1)=N(0,2)
    covariance_matrix = np.cov(data, rowvar=False) #ar0.shape = nCols x nCols
    return covariance_matrix


def generate_random_block_covariance(number_factors: int,
                                     number_blocks: int,
                                     minimum_block_size:int =1,
                                     sigma:float =1.,
                                     random_state=None) -> np.ndarray:
    """ Generate a random block covariance matrix
    Args:
        number_factors: number of factors building the covariance matrix
        number_blocks: number of blocks in the matrix
        minimum_block_size: minimum size of each block
        sigma: variance of the noise in the covariance blocks
        random_state: random state
    Returns:
        covariance_matrix: the generated covariance matrix

    """
    rng = check_random_state(random_state)
    parts = rng.choice(range(1, number_factors - (minimum_block_size - 1) * number_blocks), number_blocks - 1, replace=False)
    parts.sort()
    parts = np.append(parts, number_factors - (minimum_block_size - 1) * number_blocks)  # add nCols to list of parts, unless minBlockSize>1
    parts = np.append(parts[0], np.diff(parts)) - 1 + minimum_block_size
    print("block sizes:" + str(parts))
    covariance_matrix = None
    for number_factors_ in parts:
        # number of observations the underlying factors are generated from
        number_observationsـ = int(max(number_factors_ * (number_factors_ + 1) / 2., 100))
        covariance_ = generate_sub_covariance(number_observationsـ,
                                               number_factors_,
                                               sigma,
                                               random_state=rng)
        if covariance_matrix is None:
            covariance_matrix = covariance_.copy()
        else:
            covariance_matrix = block_diag(covariance_matrix, covariance_)  # list of square matrix on larger matrix on the diagonal

    return covariance_matrix


def generate_random_block_correlation_matrix(number_factors: int,
                                             number_blocks: int ,
                                             minimum_block_size: int=1,
                                             random_state=None
                                             ):
    """Generate a random block correlation matrix
    Args:
        number_factors: number of factors building the covariance matrix
        number_blocks: number of blocks in the matrix
        minimum_block_size: minimum size of each block
        sigma: variance of the noise in the covariance blocks
        random_state: random state
    Returns:
        covariance_matrix: the generated covariance matrix
    """
    rng = check_random_state(random_state)

    covariance_signal = generate_random_block_covariance(number_factors,
                                            number_blocks,
                                            minimum_block_size=minimum_block_size,
                                            sigma=.5,
                                            random_state=rng)
    # generate noise
    covariance_noise = generate_random_block_covariance(number_factors,
                                            1,
                                            minimum_block_size=minimum_block_size,
                                            sigma=1.,
                                            random_state=rng)
    # add noise
    covariance_signal += covariance_noise
    correlation_matrix = covariance2correlation(covariance_signal)
    correlation_df = pd.DataFrame(correlation_matrix)
    return correlation_df






def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10, debug=False):
    corr0[corr0 > 1] = 1
    dist_matrix = ((1 - corr0.fillna(0)) / 2.) ** .5
    silh_coef_optimal = pd.Series(dtype='float64')  # observations matrixs
    kmeans, stat = None, None
    maxNumClusters = min(maxNumClusters, int(np.floor(dist_matrix.shape[0] / 2)))
    print("maxNumClusters" + str(maxNumClusters))
    for init in range(0, n_init):
        # The [outer] loop repeats the first loop multiple times, thereby obtaining different initializations. Ref: de Prado and Lewis (2018)
        # DETECTION OF FALSE INVESTMENT STRATEGIES USING UNSUPERVISED LEARNING METHODS
        for num_clusters in range(2, maxNumClusters + 1):
            # (maxNumClusters + 2 - num_clusters) # go in reverse order to view more sub-optimal solutions
            kmeans_ = KMeans(n_clusters=num_clusters,
                             n_jobs = 5,
                             n_init=1)  # , random_state=3425) #n_jobs=None #n_jobs=None - use all CPUs
            kmeans_ = kmeans_.fit(dist_matrix)
            silh_coef = silhouette_samples(dist_matrix, kmeans_.labels_)
            stat = (silh_coef.mean() / silh_coef.std(), silh_coef_optimal.mean() / silh_coef_optimal.std())

            # If this metric better than the previous set as the optimal number of clusters
            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh_coef_optimal = silh_coef
                kmeans = kmeans_
                if debug == True:
                    print(kmeans)
                    print(stat)
                    silhouette_avg = silhouette_score(dist_matrix, kmeans_.labels_)
                    print("For n_clusters =" + str(num_clusters) + "The average silhouette_score is :" + str(
                        silhouette_avg))
                    print("********")

    newIdx = np.argsort(kmeans.labels_)
    # print(newIdx)

    corr1 = corr0.iloc[newIdx]  # reorder rows
    corr1 = corr1.iloc[:, newIdx]  # reorder columns

    clstrs = {i: corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in
              np.unique(kmeans.labels_)}  # cluster members
    silh_coef_optimal = pd.Series(silh_coef_optimal, index=dist_matrix.index)

    return corr1, clstrs, silh_coef_optimal


def makeNewOutputs(corr0, clstrs, clstrs2):
    clstrsNew, newIdx = {}, []
    for i in clstrs.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs[i])

    for i in clstrs2.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs2[i])

    newIdx = [j for i in clstrsNew for j in clstrsNew[i]]
    corrNew = corr0.loc[newIdx, newIdx]

    dist = ((1 - corr0.fillna(0)) / 2.) ** .5
    kmeans_labels = np.zeros(len(dist.columns))
    for i in clstrsNew.keys():
        idxs = [dist.index.get_loc(k) for k in clstrsNew[i]]
        kmeans_labels[idxs] = i

    silhNew = pd.Series(silhouette_samples(dist, kmeans_labels), index=dist.index)

    return corrNew, clstrsNew, silhNew


def clusterKMeansTop(corr0: pd.DataFrame, maxNumClusters=None, n_init=10):
    if maxNumClusters == None:
        maxNumClusters = corr0.shape[1] - 1

    corr1, clstrs, silh = clusterKMeansBase(corr0, maxNumClusters=min(maxNumClusters, corr0.shape[1] - 1),
                                            n_init=10)  # n_init)
    print("clstrs length:" + str(len(clstrs.keys())))
    print("best clustr:" + str(len(clstrs.keys())))
    # for i in clstrs.keys():
    #    print("std:"+str(np.std(silh[clstrs[i]])))

    clusterTstats = {i: np.mean(silh[clstrs[i]]) / np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatMean = sum(clusterTstats.values()) / len(clusterTstats)
    redoClusters = [i for i in clusterTstats.keys() if clusterTstats[i] < tStatMean]
    # print("redo cluster:"+str(redoClusters))
    if len(redoClusters) <= 2:
        print("If 2 or less clusters have a quality rating less than the average then stop.")
        print("redoCluster <=1:" + str(redoClusters) + " clstrs len:" + str(len(clstrs.keys())))
        return corr1, clstrs, silh
    else:
        keysRedo = [j for i in redoClusters for j in clstrs[i]]
        corrTmp = corr0.loc[keysRedo, keysRedo]
        _, clstrs2, _ = clusterKMeansTop(corrTmp, maxNumClusters=min(maxNumClusters, corrTmp.shape[1] - 1),
                                         n_init=n_init)
        print("clstrs2.len, stat:" + str(len(clstrs2.keys())))
        # Make new outputs, if necessary
        dict_redo_clstrs = {i: clstrs[i] for i in clstrs.keys() if i not in redoClusters}
        corrNew, clstrsNew, silhNew = makeNewOutputs(corr0, dict_redo_clstrs, clstrs2)
        newTstatMean = np.mean(
            [np.mean(silhNew[clstrsNew[i]]) / np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()])
        if newTstatMean <= tStatMean:
            print("newTstatMean <= tStatMean" + str(newTstatMean) + " (len:newClst)" + str(
                len(clstrsNew.keys())) + " <= " + str(tStatMean) + " (len:Clst)" + str(len(clstrs.keys())))
            return corr1, clstrs, silh
        else:
            print("newTstatMean > tStatMean" + str(newTstatMean) + " (len:newClst)" + str(len(clstrsNew.keys()))
                  + " > " + str(tStatMean) + " (len:Clst)" + str(len(clstrs.keys())))
            return corrNew, clstrsNew, silhNew
            # return corr1, clstrs, silh, stat



def getCovSub(nObs, nCols, sigma, random_state=None):
    #sub correl matrix
    rng = check_random_state(random_state)
    if nCols == 1:
        return np.ones((1,1))
    ar0 = rng.normal(size=(nObs, 1)) #array of normal rv
    ar0 = np.repeat(ar0, nCols, axis=1) #matrix of columns repeating rv. Simulate time-series of at least 100 elements.
    ar0 += rng.normal(loc=0, scale=sigma, size=ar0.shape) #add two rv X~Y~N(0,1), Z=X+Y~N(0+0, 1+1)=N(0,2)
    ar0 = np.cov(ar0, rowvar=False) #ar0.shape = nCols x nCols
    return ar0


def getRndBlockCov(nCols, nBlocks, minBlockSize=1, sigma=1., random_state=None):
    print("getRndBlockCov:" + str(minBlockSize))
    rng = check_random_state(random_state)
    parts = rng.choice(range(1, nCols - (minBlockSize - 1) * nBlocks), nBlocks - 1, replace=False)
    parts.sort()
    parts = np.append(parts, nCols - (minBlockSize - 1) * nBlocks)  # add nCols to list of parts, unless minBlockSize>1
    parts = np.append(parts[0], np.diff(parts)) - 1 + minBlockSize
    print("block sizes:" + str(parts))
    cov = None
    for nCols_ in parts:
        cov_ = getCovSub(int(max(nCols_ * (nCols_ + 1) / 2., 100)), nCols_, sigma, random_state=rng)
        if cov is None:
            cov = cov_.copy()
        else:
            cov = block_diag(cov, cov_)  # list of square matrix on larger matrix on the diagonal

    return cov


def randomBlockCorr(nCols, nBlocks, random_state=None, minBlockSize=1):
    # Form block corr
    rng = check_random_state(random_state)

    print("randomBlockCorr:" + str(minBlockSize))
    cov0 = getRndBlockCov(nCols, nBlocks, minBlockSize=minBlockSize, sigma=.5, random_state=rng)
    cov1 = getRndBlockCov(nCols, 1, minBlockSize=minBlockSize, sigma=1., random_state=rng)  # add noise
    cov0 += cov1
    corr0 = cov2corr(cov0)
    corr0 = pd.DataFrame(corr0)
    return corr0
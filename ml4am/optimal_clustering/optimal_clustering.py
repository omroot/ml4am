
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.utils import check_random_state
from scipy.linalg import block_diag
import matplotlib.pylab as plt
import matplotlib


from ml4am.denoising import *


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
                           #  n_jobs = 5,
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
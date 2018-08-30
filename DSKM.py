"""
Author: Ehsan Sherkat
Last modification date: August 30, 2018
"""
import numpy as np
import sys
import util

def DSKM(data, k):
    """
    Deterministic Seeding KMeans (DSKM)
    :param data
    :param k: number of clusters
    """
    initialCenters = np.zeros((k, data.shape[1]))
    initialCentersIndex = []
    initialCentersIndexOriginal = [0] * k
    data_sorted = []
    rank_documents = util.rankDocuments(data, reverse=True)

    similarities = np.zeros((data.shape[0], data.shape[0]))

    for doc in rank_documents:
        data_sorted.append(data[doc[0]])

    pairWiseSimilarityMatrix = util.pairWiseSimilarity(data_sorted)

    initialCenters[0] = data_sorted[0]
    initialCentersIndexOriginal[0] = rank_documents[0][0]
    initialCentersIndex.append(0)

    util.getDsim(similarities, pairWiseSimilarityMatrix, 0)
    averageHash = {}
    averageHash[0] = np.average(similarities[0])

    counter = 0

    while counter < k:
        max_different = 0
        for index in range(0, pairWiseSimilarityMatrix.shape[0]):
            if index not in initialCentersIndex:
                found = True
                different = 0

                for centerIndex in initialCentersIndex:
                    maxSimilarity = averageHash[centerIndex]
                    if similarities[centerIndex][index] >= maxSimilarity:
                        found = False
                        break
                    else:
                        different += 1
                        if different > max_different:
                            max_different = different
                if found:
                    initialCentersIndexOriginal[counter] = rank_documents[index][0]

                    util.getDsim(similarities, pairWiseSimilarityMatrix, index)
                    averageHash[index] = np.average(similarities[index])

                    if counter == 0:
                        initialCentersIndex[0] = index
                    else:
                        initialCentersIndex.append(index)
                    initialCenters[counter] = data_sorted[index]
                    counter += 1
                    break
        if not found:
            different = [0] * similarities.shape[0]
            for index in range(0, pairWiseSimilarityMatrix.shape[0]):
                if index not in initialCentersIndex:
                    for initialCenterIndex in initialCentersIndex:
                        different[index] += similarities[initialCenterIndex][index]
                else:
                    different[index] = sys.maxint

            index = np.argmin(different)
            initialCentersIndexOriginal[counter] = rank_documents[index][0]

            util.getDsim(similarities, pairWiseSimilarityMatrix, index)
            averageHash[index] = np.average(similarities[index])

            if counter == 0:
                initialCentersIndex[0] = index
            else:
                initialCentersIndex.append(index)
            initialCenters[counter] = data_sorted[index]

            counter += 1

    extend = 15
    for index, centerIndex in enumerate(initialCentersIndex):
        initialCenters[index] = util.getCenter(similarities[centerIndex].argsort()[-extend:][::-1], data_sorted)

    return initialCentersIndexOriginal, initialCentersIndex, initialCenters

def KMeans(data, n_clusters, seed="DSKM", conv_test=1e-6):
    """
    The original KMeans implementation optimized for text document clustering
    :param data: document-term matrix
    :param n_clusters: number of clusters
    :param seed: seeding method
    :param conv_test: conversion threshold
    :return:
    """

    inertiaMax = 0.0
    bestClustersCenter = np.zeros((n_clusters, data.shape[1]))
    bestLabels = [0] * data.shape[0]
    stepsBest = 0

    if seed == "random":
        centers = util.randomInit(data, n_clusters)
    elif seed == "DSKM":
        _, _, centers = DSKM(data, n_clusters)
    else:
        raise ValueError('Invalid seeding method. Select random or DSKM.')

    inertia_new = 1.0
    inertia = 0.0
    steps = 0

    while abs(inertia_new - inertia) > conv_test:
        steps += 1
        inertia = inertia_new
        clusters, labels, inertia_new, similarityMatrix, inertia2 = util.assignDataPoints(data, centers, n_clusters) #assignment step
        centers = util.calculateCenters(clusters, data) #update step

    if inertia > inertiaMax:
        stepsBest = steps
        inertiaMax = inertia
        bestClustersCenter = np.copy(centers)
        bestLabels = np.copy(labels)

    return bestLabels, bestClustersCenter, inertiaMax, stepsBest

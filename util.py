"""
Author: Ehsan Sherkat
Last modification date: August 30, 2018
"""
import numpy as np
import operator
from sklearn.metrics.pairwise import cosine_similarity
from nltk.cluster.util import cosine_distance

def getLabels(label_add):
    label_file = open(label_add, 'r')
    labels_true = []
    for line in label_file:
        line = line[0:1]
        labels_true.append(line)
    return labels_true

def pairWiseSimilarity(data):
    """
    pairwise similarity between all documents
    :param data: docuemnt-term matrix
    :param metric: default is 'cosine'.
    :return: pairwise similarity between all documents
    """
    return cosine_similarity(data)

def getDsim(similarities, pairWiseSimilarityMatrix, index):
    for i in range(0, pairWiseSimilarityMatrix.shape[0]):
        similarities[index][i] = 1 - cosine_distance(pairWiseSimilarityMatrix[index], pairWiseSimilarityMatrix[i])

def rankDocuments(data, reverse=True):
    """
    rank documents based on their sum tf/idf value
    :param data:
    :return:
    """
    rank = {}

    for index, doc in enumerate(data):
        rank[index] = np.sum(data[index])

    return sorted(rank.items(), key=operator.itemgetter(1), reverse=reverse)

def getCenter(docs, data_sorted):
    center = np.zeros(len(data_sorted[0]))
    sum = 0
    for doc in docs:
        center = center + data_sorted[doc]
        sum += 1
    center = center / sum

    return center

def read_document_term_matrix(path):
    """
    read document-term matrix (comma separate), tf-idf value
    :param path: path to matrix
    :param type: if it is word count of tf/idf value (count or tf)
    :return: 2D matrix (list of list)
    """
    document_term_matrix_file = open(path, 'r')
    matrix = []

    for line in document_term_matrix_file:
        line = line.replace('\r', '').replace('\n', '')
        columns = line.split(',')
        row = []
        for column in columns:
            row.append(column)
        matrix.append(row)

    return matrix


def randomInit(data, k):
    """
    The code source is: SciPy package.

    Returns k samples of a random variable which parameters depend on data.

    More precisely, it returns k observations sampled from a Gaussian random
    variable which mean and covariances are the one estimated from data.

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 are assumed to describe one
        dimensional data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.

    """
    def init_rank1(data):
        mu = np.mean(data)
        cov = np.cov(data)
        x = np.random.randn(k)
        x *= np.sqrt(cov)
        x += mu
        return x

    def init_rank_def(data):
        # initialize when the covariance matrix is rank deficient
        mu = np.mean(data, axis=0)
        _, s, vh = np.linalg.svd(data - mu, full_matrices=False)
        x = np.random.randn(k, s.size)
        sVh = s[:, None] * vh / np.sqrt(data.shape[0] - 1)
        x = np.dot(x, sVh) + mu
        return x

    nd = np.ndim(data)
    if nd == 1:
        return init_rank1(data)
    else:
        return init_rank_def(data)


def calculateCenters(clusters, data):
    """
    calculate the center of each cluster
    :param clusters: clusters
    :param data: document term matrix
    :return: centers
    """
    centers = np.zeros((len(clusters), data.shape[1]))

    for index in range(0, len(clusters)):
        sum = 0
        for doc in clusters[index]:
            centers[index] = centers[index] + data[doc]
            sum += 1
        centers[index] = centers[index] / sum

    return centers

def assignDataPoints(data, centers, k):
    """
    assign data points to the closest center
    :param data: document-term matrix
    :param centers: cluster centers
    :param k: number of clusters
    :return:
        clusters
        labels
        inertia
    """
    clusters = {}
    for index in range(0, k):
        clusters[index] = list()

    labels = [0] * data.shape[0]
    inertia = 0.0
    inertia2 = 0.0

    similarityMatrix = np.zeros((data.shape[0], k))

    for doc_index in range(0, data.shape[0]):
        max_sim = 0
        doc_label = 0
        for center_index in range(0, k):
            doc_sim = 1 - cosine_distance(data[doc_index], centers[center_index])
            similarityMatrix[doc_index, center_index] = doc_sim

            if doc_sim > max_sim:
                max_sim = doc_sim
                doc_label = center_index
        labels[doc_index] = doc_label
        inertia += max_sim
        inertia2 += (-1*max_sim+1)
        clusters[doc_label].append(doc_index)

    return clusters, labels, inertia, similarityMatrix, inertia2
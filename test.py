"""
Author: Ehsan Sherkat
Last modification date: August 30, 2018
"""
import DSKM
import numpy as np
import util
from sklearn import metrics

data = np.asarray(util.read_document_term_matrix("data/Newsgroup5"), dtype=np.float32)
labels_true = util.getLabels("data/labels")
n_clusters = 5

labels_pred, bestClustersCenter, inertiaMax, stepsBest = DSKM.KMeans(data, n_clusters, seed="DSKM")

print ("normalized_mutual_info_score: ", str(round(metrics.normalized_mutual_info_score(labels_true, labels_pred), 3)))
print ("homogeneity_score: ", str(round(metrics.homogeneity_score(labels_true, labels_pred), 3)))
print ("adjusted_rand_score: ", str(round(metrics.adjusted_rand_score(labels_true, labels_pred), 3)))

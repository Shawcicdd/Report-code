from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random
import numpy as np
means = [[3, 0], [1, 1], [1, 2],[4,7]]
cov = [[1, 0], [0, 1]]
N = 1247
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X3=np.random.multivariate_normal(means[3], cov, N)
X = np.concatenate((X0, X1, X2,X3), axis = 0)#concatenate through columns
K = 4
original_label = np.asarray([0]*N + [1]*N + [2]*N+[3]*N).T
def random_centroids(X, k):
  return X[np.random.choice(X.shape[0], k, replace=False)]
def choose_nearst_centroid(X, centroids):#return list each index whose nearest centroid  for each point
  dis = cdist(X, centroids)
  return np.argmin(dis, axis = 1)#axis=0:  min of  each column and axis = 1: min of each row, of course, this return index
def condition_stop(centroids, new_centroid):#condition coverged
  check1=[]
  for point in centroids:
    check1.append(tuple(point))
  check2=[]
  for point in new_centroid:
    check2.append(tuple(point))
  if set(check2)==set(check1):
    return True
  else:
    return False
def new_centroids(X, labels, K):
  centroids = np.zeros((K, X.shape[1]))
  for k in range(K):
    Xk = X[labels == k, :]#choose all point in k-th cluster
    centroids[k] = np.mean(Xk, axis = 0) 
  return centroids
def kmeans(X, K):
  centroids = [random_centroids(X, K)]
  labels = []
  it = 0
  while it<20:
    labels.append(choose_nearst_centroid(X, centroids[-1]))#centroids[-1] là bộ centroid được cập nhật ở ngay bước gần nhất
    new_centroid = new_centroids(X, labels[-1], K)
    if condition_stop(centroids[-1], new_centroid):
      break
    centroids.append(new_centroid)
    it += 1
  return (centroids, labels, it)
(centroids, labels, it) = kmeans(X, K)
print('Tọa độ các centroids:\n', centroids[-1])
label = labels[-1]# choose list label of all data nearest
plt.scatter([X[i][0] for i in range(len(X)) if label[i]==0], [X[i][1] for i in range(len(X)) if label[i]==0], c='r')
plt.scatter([X[i][0] for i in range(len(X)) if label[i]==1], [X[i][1] for i in range(len(X)) if label[i]==1], c='b')
plt.scatter([X[i][0] for i in range(len(X)) if label[i]==2], [X[i][1] for i in range(len(X)) if label[i]==2], c='g')
plt.scatter([X[i][0] for i in range(len(X)) if label[i]==3], [X[i][1] for i in range(len(X)) if label[i]==3], c='#1f77b4')
print("sai khác so với label ban đầu là:",np.mean(original_label==labels[-1]))

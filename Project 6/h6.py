import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd

np.random.seed(152)
def draw_confidence_ellipse(mean, cov, n_std=1.3, facecolor='none', **kwargs):
    mean, cov, ax = np.asarray(mean), np.asarray(cov), plt.gca()
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x, mean_x = np.sqrt(cov[0, 0]) * n_std, mean[0]
    scale_y, mean_y = np.sqrt(cov[1, 1]) * n_std, mean[1]
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

class Kmeans():
    def __init__(self, X, K):
        self.X, self.K = X, K
        # compute mean and cov
        mean = np.mean(X, axis=0)
        cov = np.cov(X.T)

        # set random initialization
        self.centroids = np.random.multivariate_normal(mean, cov, K)


    def cluster(self, iter=2):
        # computer cluster membership
        labels = np.array([np.argmin([np.linalg.norm(x - y) for y in self.centroids]) for x in self.X])
        if iter == 0:
            # compute the labels
            return labels
        
        # update centroids
        for i in range(self.K):
            self.centroids[i] = np.mean(self.X[labels == i], axis=0)
        return self.cluster(iter-1)

class EMClustering():
    def __init__(self, X, init_memb, K):
        self.X, self.labels = np.asarray(X), np.asarray(init_memb)
        self.K = K
        
    def cluster(self, iter=100):
        # find the means and covariances for each cluster
        means, covs, priors = [], [], []
        for i in range(self.K):
            means.append(np.mean(X[self.labels==i], axis=0))
            covs.append(np.cov(X[self.labels==i].T))
            priors.append(len(X[self.labels==i])/len(X))
        
        if iter == 0:
            return self.labels, np.asarray(means), np.asarray(covs), np.asarray(priors)

        H = np.zeros((len(self.X), self.K))
        for i in range(len(self.X)):
            for j in range(self.K):
                H[i][j] = multivariate_normal.pdf(X[i], mean=means[j], cov=covs[j]) * priors[j]
            H[i, :] /= sum(H[i, :])
        
        # update labels
        self.labels = np.array([np.argmax(H[i]) for i in range(len(self.X))])
        
        return self.cluster(iter-1)
        

means = []
# add means
means.append([2.5, 2.5])
means.append([-2.5, 2.5])
means.append([-2.5, -2.5])
means.append([2.5, -2.5])
means.append([0, 0])

covars = []
# add covariances metrices
covars.append([[0.8, -0.6], [-0.6, 0.8]])
covars.append([[0.8, 0.6], [0.6, 0.8]])
covars.append([[0.8, -0.6], [-0.6, 0.8]])
covars.append([[0.8, 0.6], [0.6, 0.8]])
covars.append([[1.6, 0.0], [0.0, 1.6]])

N = [50, 50, 50, 50, 100]
colors = ['r.', 'b.', 'g.', 'y.', 'c.']
# generate points
X = []
for mean, cov, n in zip(means, covars, N):
    X.append(np.random.multivariate_normal(mean, cov, n))
X = np.concatenate(X, axis=0)

# plotting the generated points
plt.plot(X[:,0], X[:, 1], "b.")
plt.show()

kmeans = Kmeans(X, 5)
labels = kmeans.cluster(iter=2)

# plot the clustering of k-means
# for i in range(5):
#     plt.plot(X[labels == i, 0], X[labels == i, 1], colors[i])
# plt.show()

em = EMClustering(X, labels, 5)
labels, clusters_means, clusters_covs, _ = em.cluster(iter=100)


for i in range(5):
    plt.plot(X[labels == i, 0], X[labels == i, 1], colors[i])

print(pd.DataFrame(clusters_means))

for i in range(5):
    draw_confidence_ellipse(means[i], covars[i], edgecolor='black', linestyle='-')
    draw_confidence_ellipse(clusters_means[i], clusters_covs[i], edgecolor='black', linestyle='--')

# plotting centroids 
plt.plot(clusters_means[:,0], clusters_means[:,1], 'ko')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import pandas as pd

class multivariate_classification:
    """multivariate classifier
    """
    np.random.seed(2413)

    def __init__(self):
        self.w_mat, self.w_vec, self.w_scalar = [], [], []

    @staticmethod
    def generate_point(means, cov_matrices, sizes):
        """generate points from normal disributions

        Args:
            means ([array]): array of means
            cov_matrices (array): array of covariance matrices
            sizes (array): list of desired number of points corrosponding to the means, and cov_matrices.

        Returns:
            list: the gerated points
        """
        X = None
        for mn, cov, sz in zip(means, cov_matrices, sizes):
            x = np.stack(np.random.multivariate_normal(mn, cov, sz).T, axis = 1)
            X = x if X is None else np.concatenate((X, x))
        return X
    
    def sample_mean(self, X, Y):
        means = []
        k = max(Y)
        for i in range(k+1):
            x1, x2 = X[Y == i][:, 0], X[Y == i][:, 1]
            means.append(np.array([sum(x1)/len(x1), sum(x2)/len(x2)]))
        
        return means
    
    def sample_cov(self, X, Y):
        means = self.sample_mean(X,Y)
        covs = []
        k = max(Y)

        for i in range(k+1):
            x1, x2 = X[Y == i][:, 0], X[Y == i][:, 1]
            mean_x1, mean_x2 = means[i]
            cov_xx = sum((x1 - mean_x1)**2)/len(x1)
            cov_yy = sum((x2 - mean_x2)**2)/len(x2)
            cov_xy = sum((x1 - mean_x1) * (x2 - mean_x2))/len(x1)
            covs.append(np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]]))
        return covs
        
    def priors(self, Y):
        priors = []
        k = max(Y)
        for i in range(k+1):
            priors.append(sum(labels == i)/len(labels))
        return priors

    def fit(self, X, Y):
        """fit the data points to the model

        Args:
            X (list): array of shape (N, 2) that represent the data points features
            Y (list): array of shape(N, ) that represents the labels of the data points
        """
        means = self.sample_mean(X, Y)
        print("sample means")
        print(means)

        cov_mats = self.sample_cov(X, Y)
        print("sample covariance matrices")
        print(cov_mats)

        priors = self.priors(Y)
        print("priors")
        print(priors)

        for mean, cov, pri in zip(means, cov_mats, priors):
            cov_inv = cho_solve(cho_factor(cov), np.eye(2))
            self.w_mat.append(-1/2 * cov_inv)
            self.w_vec.append(np.matmul(cov_inv, mean))
            self.w_scalar.append(-1/2 * np.matmul(np.matmul(mean.T, cov_inv), mean)- 1/2 * np.log(np.linalg.det(cov)) + np.log(pri))

    def get_score(self, x):
        """returns the score of x for each class

        Args:
            x (list): list of shape (D,)

        Returns:
            list: list of shape (K,) that represtns the resulted socres 
        """
        scores = []
        x = np.array(x)
        for w_m, w_v, w_s in zip(self.w_mat, self.w_vec, self.w_scalar):
            scores.append(np.matmul(np.matmul(x.T, np.array(w_m)), x) + np.matmul(np.array(w_v).T , x) + np.array(w_s))

        return scores
    
    def predict(self, X):
        predictions = []
        for x in X:
            scores = self.get_score(x)
            predictions.append(np.argmax(scores))
        return predictions
    
    def draw_boundries(self, X, Y):
        """draw the desition boundires of the model

        Args:
            X (list): training data points to be plotted
            Y (list): labels
        """
        colors = ['r.', 'g.', 'b.']
        k = max(Y)
        preds = self.predict(X)
        
        # plot the generated points
        plt.xlabel('x1')
        plt.ylabel('x2')
        
        for c in range(k+1):
            x = X[preds == c]
            plt.plot(x[:,0], x[:,1], colors[c])

        # circle the incorrect points
        plt.plot(X[preds != Y, 0], X[preds != Y, 1], "ko", markersize=13, fillstyle="none")
        
        mn_x1, mx_x1 = min(X[:, 0]), max(X[:, 0])
        mn_x2, mx_x2 = min(X[:, 1]), max(X[:, 1])
        x1 = np.linspace(mn_x1, mx_x1, 512)
        x2 = np.linspace(mn_x2, mx_x2, 512)
        grid_1, grid_2 = np.meshgrid(x1, x2)
        x1, x2 = grid_1.flatten(), grid_2.flatten()

        grid_scores = np.array([self.get_score([grid_1[i][j], grid_2[i][j]]) for i in range(len(grid_1)) for j in range(len(grid_1[0]))])
        grid_scores = grid_scores.reshape((len(grid_1), len(grid_1[0]), 3))
        A, B, C = grid_scores[:, :, 0], grid_scores[:, :, 1], grid_scores[:, :, 2]
        A[(A < B) | (A < C)] = np.nan
        B[(B < A) | (B < C)] = np.nan
        C[(C < B) | (C < A)] = np.nan
        grid_scores[:, :, 0], grid_scores[:, :, 1], grid_scores[:, :, 2] = A, B, C

        plt.contourf(grid_1, grid_2, A, levels=0, colors = 'tab:pink')
        plt.contourf(grid_1, grid_2, B, levels=0, colors = 'lime')
        plt.contourf(grid_1, grid_2, C, levels=0, colors = 'tab:cyan')
        plt.show()



means = [[0.0, 2.5], [-2.5, -2.0], [2.5, -2.0]]
cov_matrices = [[[3.2, 0.0], [0.0, 1.2]], [[1.2, -0.8], [-0.8, 1.2]], [[1.2, 0.8], [0.8, 1.2]]]
sizes = [120, 90, 90]
labels = np.concatenate([np.repeat(0, sizes[0]), np.repeat(1, sizes[1]), np.repeat(2, sizes[2])])
colors = ['r.', 'g.', 'b.']

# obtain sample points
X = multivariate_classification.generate_point(means, cov_matrices, sizes)

# plot the generated points
for (x, y), l in zip(X, labels):
    plt.plot(x, y, colors[l], markersize = 10)
# plt.show()

classifier = multivariate_classification()

# fit the data 
classifier.fit(X, labels)

# obtain preidictions
predictions = classifier.predict(X)

confusion_matrix = pd.crosstab([predictions], [labels], rownames=['y_prediction'], colnames = ['y_truth'])
print(confusion_matrix)

# draw the boundries
classifier.draw_boundries(X, labels)

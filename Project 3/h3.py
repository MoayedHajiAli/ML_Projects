import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import random

class Letter_Classifier:

    def sigmoid(self, x):
        return (1 / (1 + np.e**(-x)))

    def safelog(self, x):
        return np.log(x + 1e-100)

    @staticmethod
    def visualize_letter(x, img, x_dim=16, y_dim=20):
        x = x.reshape((x_dim, y_dim)).T
        img.imshow(1 - x, cmap="gray", interpolation="nearest")

    @staticmethod
    def split_train_test(X, Y, train_n):
        test_inds =  np.linspace(0, len(X)-1, len(X), dtype=int)
        train_inds = random.sample(range(len(X)), train_n)
        test_inds = np.delete(test_inds, train_inds)

        return X[train_inds], Y[train_inds], X[test_inds], Y[test_inds]


    def visulaize_params(self):
        _, imgs = plt.subplots(1, 5)
        for i, p in enumerate(self.P):
            self.visualize_letter(p, imgs[i])
        plt.show()

    def fit(self, X, Y):
        self.X, self.K, self.N, self.D = X, np.max(Y) + 1, len(X), len(X[0])
        self.P = np.zeros((self.K,self.D))
        self.priors = np.zeros((self.K, ))
        cnt = np.zeros((self.K,))
        for x, y in zip(X, Y):
            self.P[y] += x
            cnt[y] += 1

        for i in range(self.K):
            self.P[i] /= cnt[i]
            self.priors[i] = cnt[i] / self.N
        
    def _score(self, X):
        score = []
        for x in X:
            tmp = []
            for i in range(self.K):
                tmp.append(sum([x[j] * self.safelog(self.P[i][j]) + (1 - x[j]) * self.safelog(1 - self.P[i][j]) for j in range(len(x))]) + self.priors[i])
            score.append(np.array(tmp))
        score = np.array(score)
        return score

    def predict(self, X):
        return np.argmax(self._score(X), axis=1)
    
    def conf_matrix(self, X, Y):
        return pd.crosstab(self.predict(X), Y)


# load data
X = np.genfromtxt("hw03_data_set_images.csv", delimiter=',')
Y = np.genfromtxt("hw03_data_set_labels.csv", dtype=str)

# convert labels
Y = np.array([ord(y[1]) - ord('A') for y in Y])
classifier = Letter_Classifier()

X_train, Y_train, X_test, Y_test = None, None, None, None

for i in range(max(Y)+1):
    x_tr, y_tr, x_ts, y_ts = Letter_Classifier.split_train_test(X[i * 39:(i+1)*39], Y[i * 39:(i+1)*39], train_n=25)
    X_train = x_tr if X_train is None else np.vstack([X_train, x_tr])
    Y_train = y_tr if Y_train is None else np.append(Y_train, y_tr)
    X_test = x_ts if X_test is None else np.vstack([X_test, x_ts])
    Y_test = y_ts if Y_test is None else np.append(Y_test, y_ts)

classifier.fit(X_train, Y_train)
classifier.visulaize_params()

# print confusion matrices for training data
print("Training data confusion matrix")
print(classifier.conf_matrix(X_train, Y_train))

# print confusion matrices 
print("Testing data confusion matrix")
print(classifier.conf_matrix(X_test, Y_test))




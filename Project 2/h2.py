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
    def visualize_letter(x, x_dim=20, y_dim=16):
        x = x.reshape((x_dim, y_dim))
        img = Image.fromarray(x)
        img.show()

    @staticmethod
    def split_train_test(X, Y, train_n):
        test_inds =  np.linspace(0, len(X)-1, len(X), dtype=int)
        train_inds = random.sample(range(len(X)), train_n)
        test_inds = np.delete(test_inds, train_inds)

        return X[train_inds], Y[train_inds], X[test_inds], Y[test_inds]

    def _score(self, X, W, w0):
        N = len(X)
        y_pred = np.array([self.sigmoid(np.matmul(Wi, x.T)+ wi0) for x in X for Wi, wi0 in zip(W, w0)])
        y_pred = y_pred.reshape((N, self.K))
        return y_pred

    def _loss(self, X, Y, W, w0):
        loss = 0
        y_pred = self._score(X, W, w0)
        for y_p, y_t in zip(y_pred, Y):
            loss += 0.5 * sum((y_t - y_p) ** 2)

        return loss

    def grad(self, W, w0):
        # obtain y_pred 
        y_pred = self._score(self.X, W, w0)

        Wdelta = []
        for j in range(self.K):
            y_p, y_t = y_pred[:, j], self.Y[:, j]
            tmp = np.zeros((self.D,))
            for i in range(self.N):
                tmp += (y_p[i] - y_t[i]) * y_p[i] * (1 - y_p[i]) *  self.X[i].T
            Wdelta.append(tmp)

        w0delta = []
        for j in range(self.K):
            y_p, y_t = y_pred[:, j], self.Y[:, j]
            tmp = 0
            for i in range(self.N):
                tmp += (y_p[i] - y_t[i]) * y_p[i] * (1 - y_p[i])
            w0delta.append(tmp)
        
        return np.array(Wdelta), np.array(w0delta)


    def num_grad(self, W, w0, eps = 1e-3):
        Wdelta = np.zeros(W.shape)
        w0delta = np.zeros(w0.shape)

        for i in range(len(W)):
            for j in range(len(W[0])):
                W[i, j] += eps
                y1 = self._loss(self.X, self.Y, W, w0)
                W[i, j] -= 2 * eps
                y2 = self._loss(self.X, self.Y, W, w0)
                W[i, j] += eps
                Wdelta[i, j] = (y1 - y2) / (2*eps)

        for i in range(len(w0)):
            w0[i] += eps
            y1 = self._loss(self.X, self.Y, W, w0)
            w0[i] -= 2 * eps
            y2 = self._loss(self.X, self.Y, W, w0)
            w0[i] += eps
            w0delta[i] = (y1 - y2) / (2*eps)
        
        return Wdelta, w0delta

    def optimize(self, grad, eta = 0.01, eps = 1e-1):
        Wdelta, w0delta =  np.ones((self.K, self.D)), np.ones((self.K))
        objective_func = []

        while np.sqrt(np.sum(Wdelta ** 2) + np.sum(w0delta ** 2)) > eps:
            Wdelta, w0delta = self.grad(self.W, self.w0)

            self.W -= eta * Wdelta
            self.w0 -= eta * w0delta

            y_pred = self._score(self.X, self.W, self.w0)
            # print(pd.crosstab(np.argmax(y_pred, axis=1), np.argmax(self.Y, axis=1)))

            objective_func.append(-np.sum(self.Y * self.safelog(y_pred)))
            
        return objective_func

    def fit(self, X, Y):
        self.X, self.K, self.N, self.D = X, np.max(Y) + 1, len(X), len(X[0])

        #obtain a one-hot encoding
        y_truth = np.zeros((self.N, self.K))
        y_truth[range(self.N), Y] = 1
        self.Y = y_truth

        self.W = np.random.uniform(-0.01, 0.01, size=(self.K, self.D))
        self.w0 = np.random.uniform(-0.01, 0.01, size=(self.K))
        
        return self.optimize(self.grad)

    def predict(self, X):
        return np.argmax(self._score(X, self.W, self.w0), axis = 1)
    
    def conf_matrix(self, X, Y):
        return pd.crosstab(self.predict(X), Y)


# load data
X = np.genfromtxt("hw02_data_set_images.csv", delimiter=',')
Y = np.genfromtxt("hw02_data_set_labels.csv", dtype=str)

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

objective_function = classifier.fit(X_train, Y_train)
# plot the objective function
plt.plot(objective_function, "k-")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()

# print confusion matrices for training data
print("Training data confusion matrix")
print(classifier.conf_matrix(X_train, Y_train))

# print confusion matrices 
print("Testing data confusion matrix")
print(classifier.conf_matrix(X_test, Y_test))




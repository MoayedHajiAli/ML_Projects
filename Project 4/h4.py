import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import random

class Regressogram:
    def fit(self, X, Y, x_origin=0, bm=3):
        self.x_origin = x_origin
        self.bm = bm
        tmp = [(x, y) for x, y in zip(X, Y)]
        tmp = np.array(sorted(tmp))
        steps = int((tmp[-1][0] - x_origin + (bm - 1)) // bm) + 1
        self.density = np.zeros(steps)
        # find the estimated density
        t = l = 0
        for t in range(steps):
            if tmp[l][0] > (t+1) * bm + x_origin:
                continue
            r = l
            while r+1 < len(tmp) and tmp[r+1][0] < (t+1) * bm + x_origin:
                r += 1
            self.density[t] = (sum(tmp[l:r+1,1]) / (r - l + 1))
            l = r + 1
    
    def predict(self, x):
        if x < self.x_origin or (x - self.x_origin) > self.bm * len(self.density):
            return 0
        return self.density[int((x - self.x_origin) // self.bm)]
    
    def RMSE(self, X, Y):
        y_pred = np.array([self.predict(x) for x in X])
        return np.sqrt(sum((Y - y_pred) ** 2) / len(X))
            
    def plot(self, X_train, Y_train, X_test, Y_test):
        X = np.linspace(min(min(X_test), min(X_train)) - 5, max(max(X_test), max(X_train)) + 5, 500, dtype=float)
        Y = [self.predict(x) for x in X]
        fig, ax = plt.subplots()
        ax.plot(X, Y)
        ax.plot(X_test, Y_test, 'r.', label="test")
        ax.plot(X_train, Y_train, 'b.', label="training")
        ax.legend()
        plt.show()          

class Running_Mean_Smoother:
    def fit(self, X, Y, bm=3):
        self.bm = bm
        self.XX, self.YY = X, Y
    
    def unit(self, x):
        return 1 if (np.abs(x) <= 1) else 0
    
    def predict(self, x0):
        num = sum([self.unit((x0 - x)/self.bm) * y for x, y in zip(self.XX, self.YY)])
        den = sum([self.unit((x0 - x)/self.bm) for x in self.XX])
        return  num/den if den != 0 else 0 

    def RMSE(self, X, Y):
        y_pred = np.array([self.predict(x) for x in X])
        return np.sqrt(sum((Y - y_pred) ** 2) / len(X))
            
    def plot(self, X_train, Y_train, X_test, Y_test):
        X = np.linspace(min(min(X_test), min(X_train)) - 5, max(max(X_test), max(X_train)) + 5, 500, dtype=float)
        Y = [self.predict(x) for x in X]
        fig, ax = plt.subplots()
        ax.plot(X, Y)
        ax.plot(X_test, Y_test, 'r.', label="test")
        ax.plot(X_train, Y_train, 'b.', label="training")
        ax.legend()
        plt.show() 

class Kernel_Smoother:
    def fit(self, X, Y, bm=1):
        self.bm = bm
        self.X, self.Y = X, Y
    
    def kernel(self, x):
        return 1/np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2)
    
    def predict(self, x0):
        return sum([self.kernel((x0 - x)/self.bm) * y for x, y in zip(self.X, self.Y)]) / sum([self.kernel((x0 - x)/self.bm) for x in self.X])

    def RMSE(self, X, Y):
        y_pred = np.array([self.predict(x) for x in X])
        return np.sqrt(sum((Y - y_pred) ** 2) / len(X))
            
    def plot(self, X_train, Y_train, X_test, Y_test):
        X = np.linspace(min(min(X_test), min(X_train)) - 5, max(max(X_test), max(X_train)) + 5, 500, dtype=float)
        Y = [self.predict(x) for x in X]
        fig, ax = plt.subplots()
        ax.plot(X, Y)
        ax.plot(X_test, Y_test, 'r.', label="test")
        ax.plot(X_train, Y_train, 'b.', label="training")
        ax.legend()
        plt.show() 


def split_train_test(X, Y, train_n):
    test_inds = np.linspace(0, len(X)-1, len(X), dtype=int)
    train_inds = random.sample(range(len(X)), train_n)
    test_inds = np.delete(test_inds, train_inds)

    return X[train_inds], Y[train_inds], X[test_inds], Y[test_inds]

# load data
data = np.genfromtxt("hw04_data_set.csv", delimiter=',')[1:]


X = data[:,0]
Y = data[:,1]


# X_train, Y_train, X_test, Y_test = split_train_test(X, Y, 100)
X_train, Y_train, X_test, Y_test = X[:100], Y[:100], X[100:], Y[100:]
regressogram = Regressogram()
regressogram.fit(X_train, Y_train, x_origin=0, bm=3)
print("Regressogram -> RMSW is", regressogram.RMSE(X_test, Y_test), "Where h is 3")
regressogram.plot(X_train, Y_train, X_test, Y_test)

rms = Running_Mean_Smoother()
rms.fit(X_train, Y_train, bm=3)
print("Running Mean Smoother -> RMSW is", rms.RMSE(X_test, Y_test), "Where h is 3")
rms.plot(X_train, Y_train, X_test, Y_test)

kms = Kernel_Smoother()
kms.fit(X_train, Y_train, bm=1)
print("Kernel Smoother -> RMSW is", kms.RMSE(X_test, Y_test), "Where h is 1")
kms.plot(X_train, Y_train, X_test, Y_test)

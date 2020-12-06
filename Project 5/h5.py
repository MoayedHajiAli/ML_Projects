import numpy as np
import matplotlib.pyplot as plt;


class regression_tree:
    def __init__(self, X, Y, n=15):
        self.D, self.n = X.shape[1], n
        self.X, self.Y = X, Y
        self.tree = {}
        self.build_tree(0, X, Y)

    def node_error(self, pts_y):
        if len(pts_y) == 0:
            return 0
        pred = sum(pts_y)/len(pts_y)
        return sum((pts_y - pred) ** 2)
            
    def get_split(self, pts_x, pts_y, f_ind, f_val):
        left_ind, right_ind = [], []
        for i in range(len(pts_x)):
            if pts_x[i][f_ind] < f_val:
                left_ind.append(i)
            else:
                right_ind.append(i)
        return left_ind, right_ind

    def split_error(self, pts_x, pts_y, f_ind, f_val):
        if len(pts_x) == 0:
            return 0
        left_ind, right_ind = self.get_split(pts_x, pts_y, f_ind, f_val)
        return (self.node_error(pts_y[left_ind]) + self.node_error(pts_y[right_ind])) / len(pts_x)

    def build_tree(self, ind, X, Y):
        if(len(X) < self.n):
            # a terminal node
            self.tree[ind] = [np.mean(Y)]
            return
        
        # find the best split
        f_ind, f_val, mn_error  = -1, -1, 1e9
        for k in range(self.D):
            vals = sorted(X[:,k])
            for i in range(len(vals) - 1):
                tmp_error = self.split_error(X, Y, k, (vals[i] + vals[i+1]) / 2) 
                if  tmp_error < mn_error:
                    f_ind, f_val, mn_error = k, (vals[i] + vals[i+1]) / 2, tmp_error
        
        # update the tree and recurse
        self.tree[ind] = (f_ind, f_val)
        left_ind, right_ind = self.get_split(X, Y, f_ind, f_val)
        self.build_tree(ind * 2 + 1, X[left_ind], Y[left_ind])
        self.build_tree(ind * 2 + 2, X[right_ind], Y[right_ind])

    def _predict(self, x):
        ind = 0
        while True:
            # check if terminal node 
            if len(self.tree[ind]) == 1:
                return self.tree[ind][0]
            k, v = self.tree[ind]
            if x[k] < v:
                ind = ind * 2 + 1
            else:
                ind = ind * 2 + 2

    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    def RSME(self, X, Y):
        return np.sqrt(sum((Y - self.predict(X)) ** 2) / len(X))


data_set = np.genfromtxt("hw05_data_set.csv", delimiter=',')[1:]
training_x, training_y, testing_x, testing_y = data_set[:100, 0], data_set[:100, 1], data_set[100:, 0], data_set[100:, 1]
# converting the features to the format of set of features
training_x = np.array([np.array([x]) for x in training_x])
testing_x = np.array([np.array([x]) for x in testing_x])

regression = regression_tree(training_x, training_y, n=15)
print("RMSE with p = 15:", regression.RSME(testing_x, testing_y))


plt.xlabel("x")
plt.ylabel("y")
plt.plot(training_x.flatten(), training_y, "b.", label="training")
plt.plot(testing_x.flatten(), testing_y, "r.", label="testing")
plt_points = np.linspace(min(training_x) - 5, max(training_x) + 5, 300)
tmp = np.array([np.array([x]) for x in plt_points])
plt.plot(plt_points, regression.predict(tmp))
plt.legend()
plt.show()


rms_lst = []
for i in range(5, 55, 5):
    regression = regression_tree(training_x, training_y, n=i)
    rms_lst.append(regression.RSME(testing_x, testing_y))
plt.xlabel("pre-pruning size (P)")
plt.ylabel("RMSE")
plt.plot(range(5, 55, 5), rms_lst)
plt.plot(range(5, 55, 5), rms_lst, ".", markersize=10)
plt.show()
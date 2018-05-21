import scipy.io
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
from sklearn.metrics.pairwise import rbf_kernel

mat = scipy.io.loadmat('hw5_p1a.mat')
X = mat['X']

ma = scipy.io.loadmat('hw5_p1b.mat')
Y = ma['X']

################################################## a) b) #############################################
class Linear_k_means:
    def __init__(self, X, K):
        self.X = X
        self.K = K

        # FOR B) store cluster assignments after 2 iterations and convergence
        self.iterations = 0
        self.clusters = []

    # Calculates new mu as given in lecture notes
    def new_mu(self, z, k):
        numerator = np.zeros(2)
        for i in range(0, self.X.shape[0]):
            numerator += self.X[i] * z[k][i]

        d = 0
        for j in range(0, len(z[k])):
            d += z[k][j]

        return numerator / d

    def calculate(self):

        # Define z as the centriods for which the points are assigned to
        z = {}
        for i in range(0, self.K):
            z[i] = {}

        # Define mu as the centroids
        mu = []

        # Guess their centroids by taking a random point
        for i in range(0, self.K):
            index = int(round(np.random.uniform() * 10))
            mu.append(X[index])

        #centroid after 2 iteration
        mu_2 = []

        # Run until the mu's are not changed anymore
        while True:

            # Increament iterations
            self.iterations += 1

            # Create a deep copy of z to compare with if z has changed
            z_n = copy.deepcopy(z)
            mu_n = copy.deepcopy(mu)

            # Calculate the distance from every point to each mu's
            for i in range(0, self.X.shape[0]):
                p = self.X[i]

                # Define a m_short to indicate which mu that had shortest distance to point p
                mu_short = {
                    "dis": 99,
                    "index": -1
                }

                # Calculate which mu has shortest distance
                for m_i in range(0, len(mu)):
                    dis = np.linalg.norm(p-mu[m_i])
                    if dis < mu_short["dis"]:
                        mu_short["dis"] = dis
                        mu_short["index"] = m_i

                # Assign the point to the closest mu
                z[mu_short["index"]][i] = 1

                # Assign 0 to every other mu for this point
                for k in range(0, self.K):
                    if not (mu_short["index"] == k):
                        z[k][i] = 0

            # Calculate new mu's
            for k in range(0, self.K):
                mu[k] = self.new_mu(z, k)

            # Check if z's or mu's are equal is different
            if np.array_equal(z, z_n) or np.array_equal(mu, mu_n):
                break

            # FOR B) store clusters (z)
            if self.iterations == 2:
                self.clusters.append(z_n)
                mu_2 = copy.deepcopy(mu)

        # FOR B) store clusters at convergence
        self.clusters.append(z)

        # Append all vars to plot
        self.mu = mu
        self.z = z
        self.mu_2 = mu_2

        # We're done, return the clusters
        return self.clusters

    def plot_as_requested(self):
        # Plot when 2 iterations
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(221)
        ax1.set_title("B) After 2 iterations")
        colors_1 = []
        for i in range(0, len(self.clusters[0][0])):
            if self.clusters[0][0][i] == 1:
                colors_1.append("r")
            else:
                colors_1.append("b")
        ax1.scatter(self.X[:, 0], self.X[:, 1], color=colors_1)
        ax1.scatter([i[0] for i in self.mu_2], [i[1]
                                                for i in self.mu_2], marker="P")

        # Plot when convergence reached
        ax2 = fig.add_subplot(222)
        ax2.set_title("B) Final")
        colors_2 = []
        for i in range(0, len(self.clusters[1][0])):
            if self.clusters[1][0][i] == 1:
                colors_2.append("r")
            else:
                colors_2.append("b")
        ax2.scatter(self.X[:, 0], self.X[:, 1], color=colors_2)
        ax2.scatter([i[0] for i in self.mu], [i[1]
                                              for i in self.mu], marker="P")

        # plot to see the difference between after 2 iterations and after convergence

        ax4 = fig.add_subplot(223)
        ax4.set_title(
            " Points that Changed classes (Yellow if Changed, Green if unchanged ")
        colors_3 = []
        for i in range(0, len(self.clusters[1][0])):
            if ((self.clusters[0][0][i] == 1) and (self.clusters[1][0][i] == 0)) or ((self.clusters[0][1][i] == 1) and (self.clusters[1][1][i] == 0)):
                colors_3.append("y")
            else:
                colors_3.append("g")

        ax4.scatter(self.X[:, 0], self.X[:, 1], color=colors_3)

    # Plots the current state
    def plot(self, title):
        fig = plt.figure(figsize=(10, 10))
        ax2 = fig.add_subplot(222)
        ax2.set_title(title)
        colors_2 = []
        for i in range(0, len(self.clusters[1][0])):
            if self.clusters[1][0][i] == 1:
                colors_2.append("r")
            else:
                colors_2.append("b")
        ax2.scatter(self.X[:, 0], self.X[:, 1], color=colors_2)
        ax2.scatter([i[0] for i in self.mu], [i[1]
                                              for i in self.mu], marker="P")


k_means = Linear_k_means(Y, 2)
_ = k_means.calculate()
k_means.plot_as_requested()
k_means.plot("hejhej")
# Show the plots
plt.show()

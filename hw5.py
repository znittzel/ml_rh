import scipy.io
import numpy as np
import copy
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('hw5_p1a.mat')
X = mat['X']

################################################## a) b) #############################################
def k_means(X, K):
    # Calculates new mu as given in lecture notes 
    def new_mu(z,k,x):
        numerator = np.zeros(2)
        for i in range(0,len(x)):
            numerator += x[i] * z[k][i]

        d = 0
        for j in range(0,len(z[k])):
            d += z[k][j]

        return numerator / d

    # FOR B) store cluster assignments after 2 iterations and convergence
    iterations = 0
    clusters = []
    
    # Define z as the centriods for which the points are assigned to
    z = {}
    for i in range(0,K):
        z[i] = {}

    # Define mu as the centroids
    mu = []

    # Guess their centroids by taking a random point
    for i in range(0,K):
        index = int(round(np.random.uniform() * 10))
        mu.append(X[index])

    # Run until the mu's are not changed anymore
    while True:
        
        # Increament iterations
        iterations += 1

        # Create a deep copy of z to compare with if z has changed
        z_n = copy.deepcopy(z)
        mu_n = copy.deepcopy(mu)

        # Calculate the distance from every point to each mu's
        for i in range(0, len(X)):
            p = X[i]

            # Define a m_short to indicate which mu that had shortest distance to point p
            mu_short = {
               "dis": 99,
               "index": -1
            }

            # Calculate which mu has shortest distance
            for m_i in range(0,len(mu)):
                dis = np.linalg.norm(p-mu[m_i])
                if dis < mu_short["dis"]:
                    mu_short["dis"] = dis
                    mu_short["index"] = m_i

            # Assign the point to the closest mu
            z[mu_short["index"]][i] = 1

            # Assign 0 to every other mu for this point 
            for k in range(0,K):
                if not (mu_short["index"] == k):
                    z[k][i] = 0

        # Calculate new mu's
        for k in range(0,K):
            mu[k] = new_mu(z,k,X)

        # Check if z's or mu's are equal is different
        if np.array_equal(z, z_n) or np.array_equal(mu, mu_n):
            break
            
        # FOR B) store clusters (z)
        if iterations == 2:
            clusters.append(z_n)
        
    # FOR B) store clusters at convergence
    clusters.append(z)
        
    # At this point, the mu's are "optimal" (could be local optimum)
    return mu, z, clusters

# mu, z, clusters = k_means(X,2)

# # Plot when 2 iterations
# fig = plt.figure(figsize=(10,10))
# ax1 = fig.add_subplot(221)
# ax1.set_title("B) After 2 iterations")
# colors_1 = []
# for i in range(0, len(clusters[0][0])):
#     if clusters[0][0][i] == 1:
#         colors_1.append("r")
#     else:
#         colors_1.append("b")
# ax1.scatter(X[:,0],X[:,1],color=colors_1)

# # Plot when convergence reached
# ax2 = fig.add_subplot(222)
# ax2.set_title("B) Final")
# colors_2 = []
# for i in range(0, len(clusters[1][0])):
#     if clusters[1][0][i] == 1:
#         colors_2.append("r")
#     else:
#         colors_2.append("b")
# ax2.scatter(X[:,0],X[:,1],color=colors_2)

################
# Show 
#plt.show()

################################################## c) #############################################
mat = scipy.io.loadmat('hw5_p1b.mat')
Y = mat['X']

class Kernal_k_means:

    def __init__(self, data, K, sigma):
        self.X = data
        self.K = K
        self.sigma = sigma
        self.length = len(X)

        self.z = Kernal_k_means.initZ(K)

    def mu(self, x, k, Nk):
        r = 0
        l = len(z[k])
        for m in range(0, l):
            r += self.z[k][m]*self.rbf(x, self.X[m])

        return r / Nk
    
    # RBF Kernal
    def rbf(self, x1, x2):
        delta = np.linalg.norm(x1-x2)
        se = np.square(delta)
        result = np.exp(-(se)/(2*self.sigma**2))
        return result

    # Second term of the distance function defined in lecture notes
    def snd_term(self, x, Nk, k):
        return 2*self.mu(x,k,Nk)

    # Third term of the distance function defined in lecture notes
    def td_term(self, Nk, k):
        r = 0
        lt = len(z[k])
        for m in range(0, lt):
            for l in range(0, lt):
                r += self.z[k][m]*self.z[k][l]*self.rbf(self.X[m], self.X[l])

        return r / (2*Nk)

    # Sum of number of points assigned to the k'th z
    def Nk(self,k):
        return sum(self.z[k])

    # Create's K number of dictionaries to map the points
    @staticmethod
    def initZ(K):
        # Define z as the centriods for which the points are assigned to
        z = []
        for k in range(0,K):
            z.append([])
            # Assign random points to z_nk
            for n in range(0,length):
                if np.random.uniform() > 0.5:
                    z[k].append(1)
                else:
                    z[k].append(0)
        return z

    # Distance function as defined in lecture notes
    def distance(self, x, Nk, k):
        return 1 - self.snd_term(x, Nk, k) + self.td_term(Nk, k)

    # Computed kernals. (x_n, x_m): 1 or 0
    kernals = {}

    # Create the z's
    z = np.asmatrix(initZ(K))

    # while (True):

    #     # Create a deepcopy of latest z
    #     z_n = copy.deepcopy(z)

    #     # For each point in X, assign it to the center with nereast distance
    #     for i in range(0,length):
    #         x = X[i]
    #         nearest = { "k": -1, "dis": 99 }
    #         for k in range(0,K):
    #             d = distance(x, Nk(z,k), X, z, k, sigma)
    #             if d < nearest["dis"]:
    #                 nearest["k"] = k
    #                 nearest["dis"] = d
            
    #         # Assign 1 to nearest, 0 to others
    #         z[nearest["k"]][i] = 1
    #         for k_n in range(0, K):
    #             if not k_n == nearest["k"]:
    #                 z[k_n][i] = 0

    #     # Check if z is 
    #     if np.array_equal(z, z_n):
    #         break
    print(z)
    
    # Return z
    return z

z = kernal_k_means(Y,2,0.2)

# # Plot when convergence reached
# ax3 = fig.add_subplot(223)
# ax3.set_title("C)")
# colors = []
# for i in range(0, len(z[1])):
#     if z[1][i] == 1:
#         colors.append("r")
#     else:
#         colors.append("b")
# ax3.scatter(Y[:, 0], Y[:, 1], color=colors)

# plt.show()

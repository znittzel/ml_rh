import scipy.io
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys

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
fig = plt.figure(figsize=(10,10))
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
        # Will store certain computations
        self.comp = {
            "snd": {},
            "thd": []
            }
        
        self.X = data
        self.K = K
        self.sigma = sigma
        self.length = len(data)

        self.z = self.initZ()

    def hash(self,x):
        h = str(x[0])+str(x[1])
        return h

    # Create's K number of dictionaries to map the points
    def initZ(self):
        # Define z as the centriods for which the points are assigned to
        z = []
        for k in range(0, self.K):
            z.append([])
            # Assign random points to z_nk
            for n in range(0, self.length):
                if np.random.uniform() > 0.5:
                    z[k].append(1)
                else:
                    z[k].append(0)
        return np.array(z)
    
    # RBF Kernal
    def rbf(self, x1, x2):
        d = np.linalg.norm(x1-x2)
        result = np.exp(-(d**2)/(2*self.sigma**2))

        return result

    # There is a part that both 2nd and 3rd term uses: sum_i=1^N k(x, X[i])
    # Since both uses it, we explicitly define it here.
    # Returns a vector for the point x like [K(x, X_1), K(x, X_2), ..., K(x, X_n)]
    def part_snd(self, x):
        # Compyte if not already computed
        if self.hash(x) not in self.comp["snd"]:
            r = []
            for m in range(0, self.length):
                r.append(self.rbf(x, self.X[m]))

            self.comp["snd"][self.hash(x)] = np.array(r)

        return self.comp["snd"][self.hash(x)]

    # Second term of the distance function defined in lecture notes
    def snd_term(self, x, k):
        
        res = np.sum(self.z[k] * self.part_snd(x))

        return (2*res) / self.Nk(k)

    # Third term of the distance function defined in lecture notes
    def td_term(self, k):

        # Check if already computed
        if not np.any(self.comp["thd"]):
            r = []
            for m in range(0, self.length):
                # The vector for the point X[m] might already been computed before.
                # Therefore we use part_snd
                r.append(self.part_snd(X[m]))

            # Store computation
            self.comp["thd"] = np.array(r)

        # Create a 2D matrix of z_mk and z_lk.
        zk = self.z[k].T * self.z[k]

        return np.sum(zk * self.comp["thd"]) / (2*self.Nk(k))

    # Sum of number of points assigned to the k'th z
    def Nk(self,k):
        sumZk = np.sum(self.z[k])
        return sumZk

    # Distance function as defined in lecture notes
    def distance(self, x, k):
        stm = self.snd_term(x, k)
        ttm = self.td_term(k)
        return 1 - stm + ttm

    def calculate(self):

        # Compute until nothing changes anymore
        while (True):

            # Define a z to compare with
            z_n = copy.deepcopy(self.z)

            for i in range(0,self.length):
                x = self.X[i]
                s = {"index": -1, "dis": sys.maxint}
                for k in range(0, self.K):
                    d = self.distance(x,k)
                    if d < s["dis"]:
                        s["dis"] = d
                        s["index"] = k

                print(self.comp['snd'].keys())
                
                self.z[s["index"]][i] = 1

                for k in range(0, self.K):
                    if not k == s["index"]:
                        self.z[k][i] = 0

            if np.array_equal(self.z,z_n):
                break

        return self.z

km = Kernal_k_means(Y,2,0.2)
z = km.calculate()

# Plot when convergence reached
ax2 = fig.add_subplot(222)
ax2.set_title("C) Final")
colors_2 = []
for i in range(0, km.length):
    if z[1][i] == 1:
        colors_2.append("r")
    else:
        colors_2.append("b")
ax2.scatter(Y[:,0],Y[:,1],color=colors_2)

###############
# Show
plt.show()

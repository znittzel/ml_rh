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

mu, z, clusters = k_means(X,2)

# Plot when 2 iterations
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(221)
ax1.set_title("B) After 2 iterations")
colors_1 = []
for i in range(0, len(clusters[0][0])):
    if clusters[0][0][i] == 1:
        colors_1.append("r")
    else:
        colors_1.append("b")
ax1.scatter(X[:,0],X[:,1],color=colors_1)

# Plot when convergence reached
ax2 = fig.add_subplot(222)
ax2.set_title("B) Final")
colors_2 = []
for i in range(0, len(clusters[1][0])):
    if clusters[1][0][i] == 1:
        colors_2.append("r")
    else:
        colors_2.append("b")
ax2.scatter(X[:,0],X[:,1],color=colors_2)

################
# Show 
plt.show()

################################################## c) #############################################
mat = scipy.io.loadmat('hw5_p1b.mat')
Y = mat['X']

def kernal_k_means(X, K, sigma):

    # Data length
    length = len(X)
    
    # RBF Kernal
    def rbf(x1, x2, sigma):
        delta = abs(np.subtract(x1, x2))
        squaredEuclidean = np.square(delta)
        result = np.exp(-(squaredEuclidean)/(2*sigma**2))
        return result

    # Second term of the distance function defined in lecture notes
    def snd_term(x, Nk, X, z, k):
        r = 0
        l = len(z[k])
        for m in range(0, l):
            r += z[k][m]*rbf(x, X[m])

        return (2*r) / Nk

    # Third term of the distance function defined in lecture notes
    def td_term(Nk, X, z, k):
        r = 0
        lt = len(z[k])
        for m in range(0, lt):
            for l in range(0, l):
                r += z[k][m]*z[k][l]*rbf(X[m], X[l])

        return r / (2*Nk)

    # Sum of number of points assigned to the k'th z
    def Nk(z,k):
        return sum(z[k])

    # Create's K number of dictionaries to map the points
    def initZ(K):
        # Define z as the centriods for which the points are assigned to
        z = {}
        for k in range(0,K):
            z[k] = {}
            # Assign random points to z_nk
            for n in range(0,length):
                if np.random.uniform() > 0.5:
                    z[k][n] = 1
                else:
                    z[k][n] = 0
        return z

    # Distance function as defined in lecture notes
    def distance(x, Nk, data, z, k, sigma):
        return rbf(x, x, sigma) - snd_term(x, Nk, data, z, k) + td_term(Nk, data, z, k)

    # Create the z's
    z = initZ(K)

    while (True):

        # Create a deepcopy of latest z
        z_n = copy.deepcopy(z)

        # For each point in X, assign it to the center with nereast distance
        for i in len(X):
            x = X[i]
            nearest = { "k": -1, "dis": 99 }
            for k in range(0,K):
                d = distance(x, Nk(z,k))
                if d < nearest["dis"]:
                    nearest["k"] = k
            
            # Assign 1 to nearest, 0 to others
            z[nearest["k"]][i] = 1
            for k_n in range(0, K):
                if not k_n == nearest["k"]:
                    z[k_n][i] = 0

        # Check if z is 
        if np.array_equal(z, z_n):
            break


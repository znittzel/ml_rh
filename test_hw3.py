import numpy as np
import random

def get_data(features, rows):
	data = []
	labels = []
	for i in range(0,rows):
		row = []
		for j in range(0,features):
			row.append(random.uniform(0, 1))
		data.append(row)
		labels.append(random.getrandbits(1))

	return data,labels

def E(t,y):
	return 0.5*(t-y)**2

def Ep(l,t,w,x):
	x_sq = np.power(x,2)
	wx = w*x_sq
	return l*(wx-np.multiply(t,x))

# Set a w
w = np.array([random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)])

# Import and shuffle data
data = np.genfromtxt('dataset2.txt', delimiter=',')

# Shuffle the data
np.random.shuffle(data)

# Partition it into test and train data
train_data = data[0:1900]
test_data = data[1901:2000]

# Get lenght of data
train_length = len(train_data)

print("Learning...")
# Learn stuff
for i in range(0,train_length):
	l = train_data[i][3] # Label
	d = train_data[i][:-1] # Data
	w_d = Ep(1,l,w,d) # Calc delta w
	# Improve w  
	w -= w_d
print("Learnt.")

# Get length of test data
test_length = len(test_data)

print("Testing...")
# Test stuff
for j in range(0,test_length):
	r = 0 
	d = np.dot(w.T,test_data[j][0:3])
	if d > 0.5:
		r = 1
	else: 
		r = -1
	
	if r == test_data[j][3]:
		print("Correct")
	else:
		print("Incorrect")
		print(d)


import numpy as np
import pandas as pd



def get_data():
	df = pd.read_csv('ecommerce_data.csv')
	data = df.to_numpy()

	# everything but last col
	X = data[:, :-1]
	# only last col
	Y = data[:, -1]

	#Col 0 : is mobile
	#Col 1 : products viewed
	#Col 2 : duration
	#Col 3 : is returning
	#Col 4 : time

	X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
	X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

	N, D = X.shape
	X2 = np.zeros((N, D+3))
	X2[:, 0:(D-1)] = X[:,0:(D-1)]
	# ^ copy all but time col
	#now we have last 4 of new matrix 

	for n in range(N):
		t = int(X[n,D-1])
		X2[n,t+D-1]=1

	#different way 
		#to do : investigate functions
	Z = np.zeros((N,4))
	Z[np.arange(N), X[:,D-1].astype(np.int32)] =1

	assert(np.abs(X2[:,-4:] -Z).sum() < 10e-10)

	return X2, Y

def get_binary_data():
	x, y = get_data()
	x2 = x[y <= 1]
	y2 = y[y <= 1]
	return x2, y2



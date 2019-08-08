import numpy as np 
import matplotlib.pyplot as plt 

def forward(x,w1,b1,w2,b2):
	z = 1 / (1+ np.exp(-x.dot(w1)-b1))
	a=z.dot(w2)+b2
	expA = np.exp(a)
	y = expA / expA.sum(axis=1,keepdims=True)
	return y,z

def classification_rate(y,p):
	n_correct = 0
	n_total = 0
	for i in  range(len(y)):
		n_total += 1
		if y[i] == p[i]:
			n_correct += 1 
	return float(n_correct) / n_total

def cost(t,y):
	tot= t*np.log(y)
	return tot.sum()

def derivative_w2(Z,T,Y):
	return Z.T.dot(T-Y)

def derivative_b2(T, Y):
	return (T-Y).sum(axis=0)

def derivative_w1(X, Z, T, Y, w2):
	dZ = (T-Y).dot(w2.T)*Z*(1-Z)
	return X.T.dot(dZ)


def derivative_b1(T,Y,w2, Z):
	return ((T-Y).dot(w2.T)*Z*(1-Z)).sum(axis=0)


def main():
	#vars
	NClass = 500
	D = 2 #input dims
	M = 3 #hidden layer size
	K = 3 #number of classes
	MaxEpochs = 100000
	learning_rate = 10e-7
	costs = []


	#random inputs
	x1 = np.random.randn(NClass, D) + np.array([0,-2])
	x2 = np.random.randn(NClass, D) + np.array([2,2])
	x3 = np.random.randn(NClass, D) + np.array([-2,2])
	x = np.vstack([x1,x2,x3])
	y = np.array([0]*NClass + [1]*NClass + [2]*NClass)
	N = len(y)

	T = np.zeros((N,K))
	for i in range(N):
		T[i,y[i]] = 1

	plt.scatter(x[:,0], x[:,1], c=y, s=100, alpha=0.5)
	plt.show()
	#init weights and biases
	w1 = np.random.randn(D,M)
	b1 = np.random.randn(M)
	w2 = np.random.randn(M,K)
	b2 = np.random.randn(K)


	for epoch in range(MaxEpochs):
		output, hidden = forward(x,w1,b1,w2,b2)
		if epoch % 100 == 0:
			c = cost(T,output)
			p = np.argmax(output, axis=1)
			r = classification_rate(y,p)
			print ("cost: ", c, " classification_rate: ", r)
			costs.append(c)

		w2 += learning_rate * derivative_w2(hidden, T, output)
		b2 += learning_rate * derivative_b2(T, output)
		w1 += learning_rate * derivative_w1(x,hidden,T,output,w2)
		b1 += learning_rate * derivative_b1(T,output,w2,hidden)

	plt.plot(costs)
	plt.show()


if __name__ == '__main__':
	main()
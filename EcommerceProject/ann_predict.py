import numpy as np 
from process import get_data

x,y=get_data()

#Hidden units
m = 5
#input shape
d = x.shape[1]
k = len(set(y))

#initialize weights and biases
w1 = np.random.randn(d,m)
b1 = np.zeros(m)
w2 = np.random.randn(m,k)
b2 = np.zeros(k)

def softmax(a):
	expa = np.exp(a)
	return expa / expa.sum(axis=1, keepdims=True)

def forward(x, w1, b1, w2, b2):
	z = np.tanh(x.dot(w1)+b1)
	return softmax(z.dot(w2)+b2)

P_Y_given_x = forward(x,w1, b1, w2, b2)
predictions = np.argmax(P_Y_given_x, axis=1)

def classification_rate(y, p):
	return np.mean(y==p)

print ("Score:", classification_rate(y, predictions))
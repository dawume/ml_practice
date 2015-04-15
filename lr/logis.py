#!/bin/python
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(dataFile):
	dataMat = []; labelMat = []
	data = np.loadtxt(dataFile, delimiter = '\t')
	dataMat = data[:, 0:-1]
	labelMat = data[:, -1]
	return dataMat, labelMat

def sigmoid(num):
	return 1.0 / (1 + np.exp(-num))

def stocGrad(dataMat, labelMat) :
	pass

def grad(dataMat, labelMat):
	alpha = 0.001
	theta = np.random.rand(3)
	for i in np.arange(2000):
		out = sigmoid(np.dot(data, theta))
		J =  np.sum(- label * np.log(out) - (1 - label) * np.log(1 - out))
		print J
		theta_delta = np.dot(data.transpose(), out - label)
		theta = theta - alpha * theta_delta
		if i % 200 == 0:
			plotBestFit(data, label, theta)

def plotBestFit(dataMat, labelMat, weights) :
	n = np.shape(dataMat)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1 :
			xcord1.append(dataMat[i,0]); ycord1.append(dataMat[i,1])
		else :
			xcord2.append(dataMat[i,0]); ycord2.append(dataMat[i,1])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
	ax.scatter(xcord2, ycord2, s = 30, c = 'green')
	x = np.arange(-3.0, 3.0, 0.1)
	y = (- weights[2] - weights[0] * x) / weights[1]
	ax.plot(x,y)
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show()

if __name__ == '__main__' :
	dataMat, label = loadDataSet('../data/testSet.txt')
	m,n = np.shape(dataMat)
	data = np.hstack((dataMat, np.ones((m,1))))
	grad(data, label)
	

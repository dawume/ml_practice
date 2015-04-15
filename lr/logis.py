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
	alpha = 0.0001
	lamda = 0
	theta = np.random.rand(3)
	m, n = np.shape(dataMat)
	for i in np.arange(100):
		for j in np.arange(m):
			out = sigmoid(np.dot(data[j,:], theta))
			J =  np.sum(- label[j] * np.log(out) - (1 - label[j]) * np.log(1 - out)) + np.sum(theta[1:] ** 2)
			theta_delta = np.dot(data[j,:].transpose(), out - label[j]) + 2 * lamda * theta
			theta_delta[0] =  theta_delta[0] - 2 * lamda * theta[0]
			theta = theta - alpha * theta_delta

		print J
		if i % 10 == 0:
			plotBestFit(data, label, theta)

def grad(dataMat, labelMat):
	alpha = 0.001
	lamda = 10
	theta = np.random.rand(3)
	for i in np.arange(2000):
		out = sigmoid(np.dot(data, theta))
		J =  np.sum(- label * np.log(out) - (1 - label) * np.log(1 - out)) + np.sum(theta[1:] ** 2)
		theta_delta = np.dot(data.transpose(), out - label) + 2 * lamda * theta
		theta_delta[0] =  theta_delta[0] - 2 * lamda * theta[0]
		theta = theta - alpha * theta_delta

		print J
		if i % 200 == 0:
			plotBestFit(data, label, theta)

def plotBestFit(dataMat, labelMat, weights) :
	n = np.shape(dataMat)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1 :
			xcord1.append(dataMat[i,1]); ycord1.append(dataMat[i,2])
		else :
			xcord2.append(dataMat[i,1]); ycord2.append(dataMat[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
	ax.scatter(xcord2, ycord2, s = 30, c = 'green')
	x = np.arange(-3.0, 3.0, 0.1)
	y = (- weights[0] - weights[1] * x) / weights[2]
	ax.plot(x,y)
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show()

if __name__ == '__main__' :
	dataMat, label = loadDataSet('../data/testSet.txt')
	m,n = np.shape(dataMat)
	data = np.hstack((np.ones((m,1)), dataMat))
	#stocGrad(data, label)
	grad(data, label)
	

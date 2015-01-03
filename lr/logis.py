#!/bin/python
from numpy import *

def loadDataSet() :
	dataMat = []; labelMat = []
	f = open('testSet.txt')
	for line in f :
		fs = line.rstrip().split()
		dataMat.append([1.0, float(fs[0]), float(fs[1])])
		labelMat.append(int(fs[2]))
	return dataMat, labelMat

def sigmoid(num) :
	return 1.0 / (1 + exp(-num))

def grad(dataMat, labelMat) :
	dataMatrix = mat(dataMat)
	labelMatrix = mat(labelMat).transpose()
	m,n = shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = ones((n,1))
	for k in xrange(maxCycles) :
		h = sigmoid(dataMatrix * weights)
		error = (labelMatrix - h)
		weights = weights + alpha * dataMatrix.transpose() * error * sum((array(h) * (1 - array(h))))
	return weights

def stocGrad(dataMat, labelMat) :
	m, n = shape(dataMat)
	weights = ones(n)
	for k in range(200) :
		for i in range(m) :
			alpha = 4/(1.0 + k + i) + 0.01
			h = sigmoid(sum(dataMat[i] * weights))
			error = labelMat[i] - h
			weights = weights + alpha * error * h * (1-h) * array(dataMat[i])
	return mat(weights).T

def plotBestFit(wei) :
	import matplotlib.pyplot as plt
	weights = wei.getA()
	dataMat, labelMat = loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1 :
			xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
		else :
			xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = arange(-3.0, 3.0, 0.1)
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show()

if __name__ == '__main__' :
	dataMat, labelMat = loadDataSet()
	weights = grad(dataMat, labelMat)
	print weights

	# plotBestFit(weights)
	weights = stocGrad(dataMat, labelMat)
	print weights
	plotBestFit(weights)

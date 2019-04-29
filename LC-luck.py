# coding:utf-8
from __future__ import print_function
from random import shuffle
from past.builtins import xrange
import pickle
import numpy as np

# 文件读取
def unpickle(file):
	with open(file,'rb') as fo:
		dict = pickle.load(fo, encoding='latin1')
	return dict

def load_file(file):
	dictTrain = unpickle(file + "data_batch_1")
	dataTrain = dictTrain['data']
	labelTrain = dictTrain['labels']

	for i in range(2,6):
		dictTrain = unpickle(file + "data_batch_" + str(i))
		dataTrain = np.vstack([dataTrain,dictTrain['data']])
		labelTrain = np.hstack([labelTrain,dictTrain['labels']])

	dictTest = unpickle(file + "test_batch")
	dataTest = dictTest['data']
	labelTest = dictTest['labels']
	labelTest = np.array(labelTest)

	return dataTrain, labelTrain, dataTest, labelTest


#softmax loss 函数
def softmax_loss_naive(W, X, y, reg):
	'''
		W:权重矩阵
		X:图片训练集(矩阵)
		y:图片训练集标签(数组)
		reg:正则化强度

		return:
			loss:训练集平均loss值
			dW:梯度矩阵
	'''
	#初始化数据
	loss = 0.0
	dW = np.zeros_like(W)
	num_train = X.shape[0]	#样本数
	num_class = W.shape[1]	#样本类别数

	for i in xrange(num_train):
		score = X[i].dot(W)
		score -= np.max(score)	#提高样本稳定性

		correct_score = score[y[i]]
		exp_sum = np.sum(np.exp(score))
		loss += np.log(exp_sum) - correct_score

		for j in xrange(num_class):
			if (j == y[i]):
				dW[:, j] += np.exp(score[j]) / exp_sum * X[i] - X[i]
			else:
				dW[:, j] += np.exp(score[j]) / exp_sum * X[i]


	loss /= num_train
	loss += 0.5 * reg * np.sum(W*W)

	dW /= num_train
	dW += reg * W

	return loss, dW

#线性分类器
class LinearClassifier(object):
	def __init__(self):
		self.W = None

	def train(self, X, y, step_size = 1e-3, reg = 1e-5, num_iters = 100, batch_size = 200, verbose = True):
		'''
		X:图片训练集(矩阵)
		y:图片训练集标签(数组)
		step_size:学习步进速度
		reg:正则化强度
		num_iters:迭代次数
		batch_size:每次迭代图片样本数
		verbose:是否打印信息

		return:
			loss_history:每次训练loss值
		'''
		num_train, dim = X.shape
		num_classes = np.max(y) + 1
	
		if self.W is None:
			self.W = 0.001 * np.random.randn(dim, num_classes)

		loss_history = []

		for it in xrange(num_iters):
			#从样本中不重复随机采batch_size个样本
			sample_index = np.random.choice(num_train, batch_size, replace=False)

			X_batch = X[sample_index, :]
			y_batch = y[sample_index]

			loss, grad = self.loss(X_batch, y_batch, reg)
			loss_history.append(loss)

			self.W += -step_size * grad

			if (verbose and it %10 == 0):
				print('iteration %d / %d, samples: %d, loss: %f' % (it, num_iters, batch_size, loss))

		return loss_history

	def predict(self, X):
		'''
		X:图片训练集(矩阵)

		return:
			y_pred:标签预测值
		'''
		y_pred = np.zeros(X.shape[1])

		score = X.dot(self.W)
		y_pred = np.argmax(score, axis = 1)

		return y_pred


	def loss(self, X_batch, y_batch, reg):
		'''
		X_batch:图片训练集(矩阵)
		y_batch:图片训练集标签(数组)
		reg:正则化强度

		return:
			loss:训练集平均loss值
			dW:梯度矩阵
		'''
		return softmax_loss_naive(self.W, X_batch, y_batch, reg)


#开始训练
file_path = './input/cifar-10-batches-py/'

dataTrain, labelTrain, dataTest, labelTest = load_file(file_path)

LC = LinearClassifier()

print('start training ...')
#train(self, X, y, step_size = 1e-3, reg = 1e-5, num_iters = 100, batch_size = 200, verbose = True)
#在dataTrain中不重复随机抽取batch_size个样本，迭代训练num_iters次
loss_all = LC.train(dataTrain, labelTrain, num_iters = 8000, batch_size = 200)

print('last loss is %f' %(loss_all[-1]))
#开始预测
print('start predicting ...')
y_pred = LC.predict(dataTest)

hit = 0
for i in xrange(10000):
	if (y_pred[i] == labelTest[i]):
		hit += 1

print('the accuracy rate is %f ' % (hit/10000))









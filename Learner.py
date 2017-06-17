import tensorflow as tf
import random as rd

def RtnToObj(rtn):
    if rtn >= 0.03:
        return [1, 0]
    elif rtn <= -0.03:
        return [0, 1]
    elif rtn > -0.03 and rtn < 0.03: # -0.03 < rtn < 0.03:
        return [0, 0]
    else:
        return None

class Encoder:
    def __init__(self, inputNum, encNum, learnRate, eraseProb, initStd, costOrder):
        self.inputNum = inputNum
        self.encNum = encNum
        self.learnRate = learnRate
        self.eraseProb = eraseProb
        self.initStd = initStd
        self.costOrder = costOrder
        self.trainedNum = 0
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder('float', [None, self.inputNum], name = 'X')
            self.Y = tf.placeholder('float', [None, self.inputNum], name = 'Y')
            wEnc = tf.Variable(tf.truncated_normal([self.inputNum, self.encNum], stddev = self.initStd), name = 'wEnc')
            bEnc = tf.Variable(tf.zeros([self.encNum]), name = 'bEnc')
            wDec = tf.transpose(wEnc)
            bDec = tf.Variable(tf.zeros([self.inputNum]), name = 'bDec')
            self.xEnc = tf.nn.tanh(tf.matmul(self.X, wEnc) + bEnc)
            self.xDec = tf.nn.sigmoid(tf.matmul(self.xEnc, wDec) + bDec)
            cost = tf.norm(self.Y - self.xDec, ord = self.costOrder, axis = 1, keep_dims = False)
            self.accuracy = tf.reduce_mean(tf.reduce_mean(tf.abs(self.Y - self.xDec), axis = 1, keep_dims = False), axis = 0, keep_dims = False)
            self.train = tf.train.GradientDescentOptimizer(self.learnRate).minimize(cost)
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())

    def Train(self, sample):
        acc = self.accuracy.eval(feed_dict = {self.X: sample, self.Y: sample}, session = self.sess)
        self.sess.run(self.train, feed_dict = {self.X: sample, self.Y: sample})
        self.trainedNum += len(sample)
        return acc

    def Eval(self, sample):
        return self.xEnc.eval(feed_dict = {self.X: sample}, session = self.sess)

    def TestAcc(self, sample):
        return self.accuracy.eval(feed_dict = {self.X: sample, self.Y: sample}, session = self.sess)

class ProbLearner:
    def __init__(self, inputNum, learnRate, initStd, costOrder):
        self.inputNum = inputNum
        self.outputNum = 2
        self.learnRate = learnRate
        self.initStd = initStd
        self.costOrder = costOrder
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder('float', [None, self.inputNum], name = 'X')
            self.Y = tf.placeholder('float', [None, self.outputNum], name = 'Y')
            w = tf.Variable(tf.truncated_normal([self.inputNum, self.outputNum], stddev = self.initStd), name = 'w')
            b = tf.Variable(tf.zeros([self.outputNum]), name='b')
            self.xOut = tf.nn.sigmoid(tf.matmul(self.X, w) + b)
            cost = tf.norm(self.Y - self.xOut, ord = self.costOrder, axis = 1, keep_dims = False)
            self.accuracy = tf.reduce_mean(cost, axis = 0, keep_dims = False)
            self.train = tf.train.GradientDescentOptimizer(self.learnRate).minimize(cost)
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())

    def Train(self, sample, obj):
        acc = self.accuracy.eval(feed_dict = {self.X: sample, self.Y: obj}, session = self.sess)
        self.sess.run(self.train, feed_dict = {self.X: sample, self.Y: obj})
        return acc

    def Eval(self, sample):
        return self.xOut.eval(feed_dict = {self.X: sample}, session = self.sess)

    def TestAcc(self, sample, obj):
        return self.accuracy.eval(feed_dict = {self.X: sample, self.Y: obj}, session = self.sess)



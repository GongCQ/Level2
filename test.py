import Learner
import random as rd
import numpy as np


class Lc:
    def __init__(self):
        self.learner = Learner.Encoder(10, 8, 0.05, 0, 0.1, 2)

    def Train(self, sample):
        self.learner.Train(sample)

class Fc:
    def Train(self, sample):
        return learner2.Train(sample)
fc = Fc()
lc = Lc()
learner1 = Learner.Encoder(10, 8, 0.05, 0, 0.1, 2)
learner2 = Learner.Encoder(10, 8, 0.05, 0, 0.1, 2)
inputNum = 10
encNum = 8
sampleNum = 5000
batchNum = 10
sampleList = [None] * sampleNum
for i in range(sampleNum):
    sample = [None] * inputNum
    for j in range(inputNum - 2):
        sample[j] = rd.uniform(0, 1)
    sample[inputNum - 2] = np.mean(sample[0 : inputNum - 2])
    sample[inputNum - 1] = np.std(sample[0 : inputNum - 1])
    sampleList[i] = sample

sampleBatch = []
for i in range(sampleNum):
    sampleBatch.append(sampleList[i])
    if len(sampleBatch) >= batchNum:
        print(lc.learner.Train(sampleBatch), end = '  ')
        print(learner1.Train(sampleBatch), end = '  ')
        print(fc.Train(sampleBatch))
        sampleBatch = []
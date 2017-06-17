import numpy as np
import datetime as dt
import Learner
import random as rd
import os

class Stats:
    def __init__(self, date):
        self.date = date
        self.sampleList = []
        self.actualUp = 0
        self.actualDown = 0
        self.actualMid = 0
        self.predictUp = 0
        self.predictDown = 0
        self.total = 0
        self.aupu = 0
        self.aupd = 0
        self.adpu = 0
        self.adpd = 0
        self.top5 = 0
        self.top10 = 0
        self.top20 = 0
        self.top50 = 0
        self.bot5 = 0
        self.bot10 = 0
        self.bot20 = 0
        self.bot50 = 0
        self.top5d = 0
        self.top10d = 0
        self.top20d = 0
        self.top50d = 0
        self.bot5d = 0
        self.bot10d = 0
        self.bot20d = 0
        self.bot50d = 0

    def AppendData(self, date, symbol, rtn, prob):
        if date != self.date:
            raise Exception('date error!')
        obj = Learner.RtnToObj(rtn)
        if obj is None or np.isnan(prob[0] - prob[1]):
            return
        self.total += 1
        if obj[0] == 1 and prob[0] > prob[1]:
            self.aupu += 1
        elif obj[0] == 1 and prob[0] < prob[1]:
            self.aupd += 1
        elif obj[1] == 1 and prob[0] > prob[1]:
            self.adpu += 1
        elif obj[1] == 1 and prob[0] < prob[1]:
            self.adpd += 1
        if obj[0] == 1:
            self.actualUp += 1
        elif obj[1] == 1:
            self.actualDown += 1
        else:
            self.actualMid += 1
        if prob[0] > prob[1]:
            self.predictUp += 1
        elif prob[0] < prob[1]:
            self.predictDown += 1
        self.sampleList.append([date, symbol, prob[0], prob[1], prob[0] - prob[1], obj[0], obj[1], rtn])

    def Eval(self):
        probArr = np.nan * np.zeros([len(self.sampleList)])
        for i in range(len(probArr)):
            probArr[i] = self.sampleList[i][4]
        probSortIndex = np.argsort(-probArr)
        for i in range(len(probArr)):
            seq = probSortIndex[i]
            sample = self.sampleList[seq]

            if i < 5 and sample[5] == 1:
                self.top5 += 1 / 5
            if i < 10 and sample[5] == 1:
                self.top10 += 1 / 10
            if i < 20 and sample[5] == 1:
                self.top20 += 1 / 20
            if i < 50 and sample[5] == 1:
                self.top50 += 1 / 50
            if i >= len(probArr) - 5 and sample[5] == 1:
                self.bot5 += 1 / 5
            if i >= len(probArr) - 10 and sample[5] == 1:
                self.bot10 += 1 / 10
            if i >= len(probArr) - 20 and sample[5] == 1:
                self.bot20 += 1 / 20
            if i >= len(probArr) - 50 and sample[5] == 1:
                self.bot50 += 1 / 50

            if i < 5 and sample[6] == 1:
                self.top5d += 1 / 5
            if i < 10 and sample[6] == 1:
                self.top10d += 1 / 10
            if i < 20 and sample[6] == 1:
                self.top20d += 1 / 20
            if i < 50 and sample[6] == 1:
                self.top50d += 1 / 50
            if i >= len(probArr) - 5 and sample[6] == 1:
                self.bot5d += 1 / 5
            if i >= len(probArr) - 10 and sample[6] == 1:
                self.bot10d += 1 / 10
            if i >= len(probArr) - 20 and sample[6] == 1:
                self.bot20d += 1 / 20
            if i >= len(probArr) - 50 and sample[6] == 1:
                self.bot50d += 1 / 50

        self.aupu /= max([self.actualUp, 1])
        self.aupd /= max([self.actualUp, 1])
        self.adpu /= max([self.actualDown, 1])
        self.adpd /= max([self.actualDown, 1])
        self.actualUp /= max([self.total, 1])
        self.actualDown /= max([self.total, 1])
        self.actualMid /= max([self.total, 1])
        predictTotal = max([self.predictUp + self.predictDown, 1])
        self.predictUp /= predictTotal
        self.predictDown /= predictTotal

    def ToFile(self, path):
        file = open(os.path.join(path, self.date.strftime('%Y-%m-%d') + ' sample.csv'), 'w')
        file.write('date, symbol, upProb, downProb, upProb-downProb, obj0, obj1, rtn \n')
        for sample in self.sampleList:
            file.write(str(sample[0]) + ',' + sample[1] + ',' + str(sample[2]) + ',' + str(sample[3]) + ',' +
                            str(sample[4]) + ',' + str(sample[5]) + ',' + str(sample[6]) + ',' + str(sample[7]) + '\n')
        file.flush()


class Stock:
    def __init__(self, code, symbol, name, mktCode, capacity):
        self.code = code
        self.symbol = symbol
        self.name = name
        self.mktCode = mktCode
        self.firstDay = dt.datetime.max
        self.capacity = capacity
        self.dateList = [None] * capacity
        self.openPriceList = [np.nan] * capacity
        self.closePriceList = [np.nan] * capacity
        self.l2SampleList = [None] * capacity
        self.objList = [None] * capacity
        self.latestProb = [np.nan, np.nan]
        for i in range(capacity):
            self.objList[i] = [np.nan] * 4
        self.loc = -1
        self.probList = [None] * capacity
        self.locProb = -1

    def AppendData(self, date, openPrice, closePrice, l2Sample):
        self.loc = (self.loc + 1) % self.capacity
        self.dateList[self.loc] = date
        self.openPriceList[self.loc] = openPrice
        self.closePriceList[self.loc] = closePrice
        self.l2SampleList[self.loc] = l2Sample
        self.objList[self.loc] = [np.nan, np.nan, np.nan, np.nan]
        self.objList[(self.loc - 1) % self.capacity][0] = closePrice / self.closePriceList[(self.loc - 1) % self.capacity] - 1
        self.objList[(self.loc - 2) % self.capacity][1] = closePrice / self.openPriceList[(self.loc - 1) % self.capacity] - 1
        self.objList[(self.loc - 3) % self.capacity][2] = closePrice / self.openPriceList[(self.loc - 2) % self.capacity] - 1
        self.objList[(self.loc - 5) % self.capacity][3] = closePrice / self.openPriceList[(self.loc - 4) % self.capacity] - 1
        if np.isfinite(closePrice) and date < self.firstDay:
            self.firstDay = date

    def AppendProb(self, date, prob):
        self.locProb = (self.locProb + 1) % self.capacity
        self.probList[self.locProb] = [date, prob[0], prob[1]]

    def GetSampleObj(self, backIndex, objSeq):
        return self.l2SampleList[(self.loc + backIndex) % self.capacity], \
               Learner.RtnToObj(self.objList[(self.loc + backIndex) % self.capacity][objSeq]), \
               self.dateList[(self.loc + backIndex) % self.capacity]

    def GetProb(self, backIndex):
        return self.probList[(self.locProb + backIndex) % self.capacity]


class DataAdapter:
    def __init__(self, codeList, csMap):
        self.stockList = [None] * len(codeList)
        for i in range(len(codeList)):
            self.stockList[i] = Stock(codeList[i], csMap.GetSymbol(codeList[i]), csMap.GetNameByCode(codeList[i]), csMap.GetMktByCode(codeList[i]), 125)

    def NewDayHandler(self, mkt):
        for stock in self.stockList:
            l2Sample = mkt.dsList[1].GetRecord(stock.code)
            price = mkt.dsList[0].GetRecord(stock.code)
            if price is not None:
                openPrice = price[6]
                closePrice = price[3]
            else:
                openPrice = np.nan
                closePrice = np.nan
            if l2Sample is not None:  # 归一化
                min0 = np.Inf
                max0 = -np.Inf
                min1 = np.Inf
                max1 = -np.Inf
                min2 = np.Inf
                max2 = -np.Inf
                min3 = np.Inf
                max3 = -np.Inf
                for i in range(48):
                    if np.isnan(l2Sample[6 + 4 * i + 0] + l2Sample[6 + 4 * i + 1] + l2Sample[6 + 4 * i + 2] + l2Sample[6 + 4 * i + 3]): # 包含无效数据
                        l2Sample = None
                        break
                    min0 = l2Sample[6 + 4 * i + 0] if l2Sample[6 + 4 * i + 0] < min0 else min0
                    max0 = l2Sample[6 + 4 * i + 0] if l2Sample[6 + 4 * i + 0] > max0 else max0
                    min1 = l2Sample[6 + 4 * i + 1] if l2Sample[6 + 4 * i + 1] < min1 else min1
                    max1 = l2Sample[6 + 4 * i + 1] if l2Sample[6 + 4 * i + 1] > max1 else max1
                    min2 = l2Sample[6 + 4 * i + 2] if l2Sample[6 + 4 * i + 2] < min2 else min2
                    max2 = l2Sample[6 + 4 * i + 2] if l2Sample[6 + 4 * i + 2] > max2 else max2
                    min3 = l2Sample[6 + 4 * i + 3] if l2Sample[6 + 4 * i + 3] < min3 else min3
                    max3 = l2Sample[6 + 4 * i + 3] if l2Sample[6 + 4 * i + 3] > max3 else max3
                if l2Sample is not None and not (min0 == max0 or min1 == max1 or min2 == max2 or min3 == max3):
                    for i in range(48):
                        l2Sample[6 + 4 * i + 0] = (l2Sample[6 + 4 * i + 0] - min0) / (max0 - min0)
                        l2Sample[6 + 4 * i + 1] = (l2Sample[6 + 4 * i + 1] - min1) / (max1 - min1)
                        l2Sample[6 + 4 * i + 2] = (l2Sample[6 + 4 * i + 2] - min2) / (max2 - min2)
                        l2Sample[6 + 4 * i + 3] = (l2Sample[6 + 4 * i + 3] - min3) / (max3 - min3)
                else:  # 无法归一化
                    l2Sample = None
            stock.AppendData(mkt.crtDate, openPrice, closePrice, l2Sample)


class Strategy:
    def __init__(self, codeList, csMap, trainBatch):
        self.da = DataAdapter(codeList, csMap)
        self.statsDict = {}
        # 编码层参数
        self.encInputNum = 192
        self.encEncNum = 192
        self.encLearnRate = 0.05
        self.encEraseProb = 0
        self.encInitStd = 0.1
        self.encCostOrder = 2
        self.encTrainBatch = 5
        # 概率学习层参数
        self.probInputNum = self.encEncNum
        self.probOutputNum = 2
        self.probLearnRate = 0.05
        self.probInitStd = 0.1
        self.probCostOrder = np.Inf
        self.probTrainBatch = 50
        # 初始化网络
        self.ResetLearner()
        self.encoder = Learner.Encoder(self.encInputNum, self.encEncNum, self.encLearnRate, self.encEraseProb,
                                       self.encInitStd, self.encCostOrder)
        self.probLearner = Learner.ProbLearner(self.probInputNum, self.probLearnRate, self.probInitStd,
                                               self.probCostOrder)

    def ResetLearner(self):
        self.lastReset = 0
        self.trainedActive = 0
        self.trainedNegative = 0
        self.encoder = Learner.Encoder(self.encInputNum, self.encEncNum, self.encLearnRate, self.encEraseProb,
                                       self.encInitStd, self.encCostOrder)
        self.probLearner = Learner.ProbLearner(self.probInputNum, self.probLearnRate, self.probInitStd,
                                               self.probCostOrder)

    def TrainEncoder(self, sampleBatch):
        acc = self.encoder.Train(sampleBatch)
        return acc

    def TrainProb(self, sampleBatch, objBatch):
        acc = self.probLearner.Train(sampleBatch, objBatch)
        return acc

    def ReTrain(self):
        self.ResetLearner()
        stockSeq = list(range(len(self.da.stockList)))
        rd.shuffle(stockSeq)

        # 训练编码层
        sampleBatchEnc = []
        accList = []
        for s in stockSeq:
            stock = self.da.stockList[s]
            dateSeq = list(range(stock.capacity))
            rd.shuffle(dateSeq)
            for d in dateSeq:
                date = stock.dateList[d]
                sample = stock.l2SampleList[d]
                if date is None or sample is None:
                    continue
                sampleBatchEnc.append(sample[6 : 198])
                if len(sampleBatchEnc) >= self.encTrainBatch:
                    acc = self.TrainEncoder(sampleBatchEnc)
                    sampleBatchEnc.clear()
                    accList.append(np.round(acc, 6))
        if len(sampleBatchEnc) > 0:
            acc = self.TrainEncoder(sampleBatchEnc)
            sampleBatchEnc.clear()
            accList.append(np.round(acc, 6))
        print(accList[0 : 10])
        print(accList[len(accList) - 10 : len(accList)])
        accList.clear()

        # 训练概率学习层
        sampleBatch = []
        objBatch = []
        accList = []
        for s in stockSeq:
            stock = self.da.stockList[s]
            dateSeq = list(range(stock.capacity))
            rd.shuffle(dateSeq)
            for d in dateSeq:
                date = stock.dateList[d]
                obj = Learner.RtnToObj(stock.objList[d][1])
                sample = stock.l2SampleList[d]
                if date is None or sample is None or obj is None:
                    continue
                if obj[0] == 1 and self.trainedActive / (self.trainedActive + self.trainedNegative + 1) <= 0.51:
                    sampleBatch.append(sample[6 : 198])
                    objBatch.append(obj)
                    self.trainedActive += 1
                elif obj[1] == 1 and self.trainedNegative / (self.trainedActive + self.trainedNegative + 1) <= 0.51:
                    sampleBatch.append(sample[6 : 198])
                    objBatch.append(obj)
                    self.trainedNegative += 1
                if len(sampleBatch) >= self.probTrainBatch:
                    acc = self.TrainProb(self.encoder.Eval(sampleBatch), objBatch)
                    sampleBatch.clear()
                    objBatch.clear()
                    accList.append(np.round(acc, 6))
        if len(sampleBatch) > 0:
            acc = self.TrainProb(self.encoder.Eval(sampleBatch), objBatch)
            sampleBatch.clear()
            objBatch.clear()
            accList.append(np.round(acc, 6))
        print(accList[0 : 10])
        print(accList[len(accList) - 10 : len(accList)])
        accList.clear()

    def NewDayHandler(self, mkt):
        self.statsDict[mkt.crtDate] = Stats(mkt.crtDate)
        # 统计昨日预测结果
        prob = np.nan * np.zeros([len(self.da.stockList)])
        for i in range(len(self.da.stockList)):
            prob[i] = self.da.stockList[i].latestProb[0] - self.da.stockList[i].latestProb[1]

        # 训练
        if self.lastReset >= 5:
            self.ReTrain()
        self.lastReset += 1

        # 预测
        for stock in self.da.stockList:
            sample, obj, date = stock.GetSampleObj(0, 1)
            if sample is not None:
                sample = sample[6 : 198]
                sampleEnc = self.encoder.Eval([sample])
                stock.latestProb = self.probLearner.Eval(sampleEnc)[0]
            else:
                stock.latestProb = [np.nan, np.nan]
            stock.AppendProb(mkt.crtDate, stock.latestProb)

        debug = 0

    def AfterAll(self, mkt):
        for stock in self.da.stockList:
            for i in range(len(stock.dateList)):
                date = stock.dateList[i]
                if date is None:
                    continue
                self.statsDict[date].AppendData(date, stock.symbol, stock.objList[i][1], stock.probList[i][1 : 3])
        nowStr = dt.datetime.now().strftime('%Y%m%d%H%M%S')
        path = os.path.join('.', 'stats ' + nowStr)
        os.mkdir(path)
        sumFile = open(os.path.join('.', 'stats ' + nowStr, '0 summary' + '.csv'), 'w')
        sumFile.write('date,total,au,ad,am,pu,pd,aupu,aupd,adpu,adpd,t5,t10,t20,t50,b5,b10,b20,b50,t5d,t10d,t20d,t50d,b5d,b10d,b20d,b50d\n')
        for key, stats in self.statsDict.items():
            stats.Eval()
            stats.ToFile(path)
            if stats.total == 0:
                continue
            sumFile.write(str(stats.date) + ',' +
                          str(stats.total) + ',' + str(stats.actualUp) + ',' + str(stats.actualDown) + ',' +
                          str(stats.actualMid) + ',' + str(stats.predictUp) + ',' + str(stats.predictDown) + ',' +
                          str(stats.aupu) + ',' + str(stats.aupd) + ',' + str(stats.adpu) + ',' +
                          str(stats.adpd) + ',' + str(stats.top5) + ',' + str(stats.top10) + ',' +
                          str(stats.top20) + ',' + str(stats.top50) + ',' + str(stats.bot5) + ',' +
                          str(stats.bot10) + ',' + str(stats.bot20) + ',' + str(stats.bot50) + ',' +
                          str(stats.top5d) + ',' + str(stats.top10d) + ',' +
                          str(stats.top20d) + ',' + str(stats.top50d) + ',' + str(stats.bot5d) + ',' +
                          str(stats.bot10d) + ',' + str(stats.bot20d) + ',' + str(stats.bot50d) + '\n')
        sumFile.flush()
        latestProbFile = open(os.path.join(path, '0 latestProb.csv'), 'w')
        latestProbFile.write('date,symbol,upProb,downProb,upProb-downProb\n')
        for stock in self.da.stockList:
            prob = stock.GetProb(-1)
            latestProbFile.write(str(prob[0]) + ',' + stock.symbol + ',' + str(prob[1]) + ',' + str(prob[2]) + ',' + str(prob[1] - prob[2]) + '\n')

        debug = 0
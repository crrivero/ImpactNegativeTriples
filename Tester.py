# coding:utf-8
import torch
from torch.autograd import Variable
import sys
import numpy as np
from DataLoader import Triple

class Tester(object):

    def __init__(self, model=None, train_data_loader=None, valid_data_loader=None, test_data_loader=None,
                 undesired_ents=None, undesired_rels=None, use_gpu=True, corruptionMode="Global"):
        self.model = model
        self.train = train_data_loader
        self.valid = valid_data_loader
        self.test = test_data_loader
        self.undesired_ents = undesired_ents
        self.undesired_rels = undesired_rels
        self.entitySet = set(range(self.train.entityTotal))
        self.use_gpu = use_gpu
        self.corruptionMode = corruptionMode

        if self.use_gpu:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, arrH, arrR, arrT):
        return self.model.predict({
            'batch_h': self.to_var(arrH, self.use_gpu),
            'batch_r': self.to_var(arrR, self.use_gpu),
            'batch_t': self.to_var(arrT, self.use_gpu),
            'mode': 'normal'
        })

    def getCorruptedUsingDicts(self, r, hOrT, initialSet, trainDict, validDict, testDict):
        corrupted = initialSet

        if r in trainDict:
            if hOrT in trainDict[r]:
                corrupted = corrupted - trainDict[r][hOrT]
        if r in validDict:
            if hOrT in validDict[r]:
                corrupted = corrupted - validDict[r][hOrT]
        if r in testDict:
            if hOrT in testDict[r]:
                corrupted = corrupted - testDict[r][hOrT]

        return corrupted

    def unionSet(self, r, f):
        union = set()
        if r in f(self.train):
            union = union.union(f(self.train)[r])
        if r in f(self.valid):
            union = union.union(f(self.valid)[r])
        if r in f(self.test):
            union = union.union(f(self.test)[r])
        return union

    def run(self):
        testList = self.test.getTriples()

        metrics = {}
        for option in ["Desired", "Undesired", "All"]:
            metrics["pAt1_" + option] = PrecisionAtX(1)
            metrics["pAt3_" + option] = PrecisionAtX(3)
            metrics["pAt10_" + option] = PrecisionAtX(10)
            metrics["mr_" + option] = MeanRank()
            metrics["mrr_" + option] = MeanReciprocalRank()

        #materializedTriplesAll = set()
        #materializedTriplesUndesired = set()

        dom = {}
        ran = {}

        if self.corruptionMode == "TCLCWA" or self.corruptionMode == "NLCWA":
            for r in self.test.relations:
                dom[r] = self.unionSet(r, lambda d : d.getDomain())
                ran[r] = self.unionSet(r, lambda d : d.getRange())

                if self.corruptionMode == "NLCWA":
                    # All compatibles are the same for train, validation and test, they are for the whole dataset.
                    for ri in self.train.domDomCompatible[r]:
                        dom[r] = dom[r].union(self.unionSet(ri, lambda d : d.getRange()))
                    for rj in self.train.domRanCompatible[r]:
                        dom[r] = dom[r].union(self.unionSet(rj, lambda d : d.getDomain()))

                    for ri in self.train.ranRanCompatible[r]:
                        ran[r] = ran[r].union(self.unionSet(ri, lambda d : d.getDomain()))
                    for rj in self.train.ranDomCompatible[r]:
                        ran[r] = ran[r].union(self.unionSet(rj, lambda d : d.getRange()))

        totalCorruptedTriples = 0
        pending = len(testList)
        resortedToLCWA = 0
        for t in testList:
            pending = pending - 1
            corruptedHeads = set()
            corruptedTails = set()

            if self.corruptionMode=="Global":
                corruptedHeads = self.entitySet - self.train.getHeadEntities().union(
                    self.valid.getHeadEntities().union(self.test.getHeadEntities()))
                corruptedTails = self.entitySet - self.train.getTailEntities().union(
                    self.valid.getTailEntities().union(self.test.getTailEntities()))
            if self.corruptionMode == "LCWA":
                corruptedHeads = self.getCorruptedUsingDicts(t.r, t.t, self.entitySet,
                                    self.train.getHeadDict(), self.valid.getHeadDict(), self.test.getHeadDict())
                corruptedTails = self.getCorruptedUsingDicts(t.r, t.h, self.entitySet,
                                    self.train.getTailDict(), self.valid.getTailDict(), self.test.getTailDict())
            if self.corruptionMode == "TCLCWA" or self.corruptionMode == "NLCWA":
                corruptedHeads = self.getCorruptedUsingDicts(t.r, t.t, dom[t.r],
                                    self.train.getHeadDict(), self.valid.getHeadDict(), self.test.getHeadDict())
                corruptedTails = self.getCorruptedUsingDicts(t.r, t.h, ran[t.r],
                                    self.train.getTailDict(), self.valid.getTailDict(), self.test.getTailDict())
                # If heads or tails are empty, use LCWA
                if len(corruptedHeads) == 0:
                    corruptedHeads = self.getCorruptedUsingDicts(t.r, t.t, self.entitySet,
                                    self.train.getHeadDict(), self.valid.getHeadDict(), self.test.getHeadDict())
                    resortedToLCWA = resortedToLCWA + 1
                if len(corruptedTails) == 0:
                    corruptedTails = self.getCorruptedUsingDicts(t.r, t.h, self.entitySet,
                                    self.train.getTailDict(), self.valid.getTailDict(), self.test.getTailDict())
                    resortedToLCWA = resortedToLCWA + 1

            if t.h in corruptedHeads:
                print("h was in set!")
                sys.exit(-1)
            if t.t in corruptedTails:
                print("t was in set!")
                sys.exit(-1)

            totalCorruptedTriples = totalCorruptedTriples + len(corruptedHeads) + len(corruptedTails)
            totalTriples = 1 + len(corruptedHeads) + len(corruptedTails)
            arrH = np.zeros(totalTriples, dtype=np.int64)
            arrR = np.zeros(totalTriples, dtype=np.int64)
            arrT = np.zeros(totalTriples, dtype=np.int64)

            current = 0
            arrH[current] = t.h
            arrR[current] = t.r
            arrT[current] = t.t
            current = current + 1

            for hPrime in corruptedHeads:
                arrH[current] = hPrime
                arrR[current] = t.r
                arrT[current] = t.t
                current = current + 1
            corruptedHeadsEnd = current
            for tPrime in corruptedTails:
                arrH[current] = t.h
                arrR[current] = t.r
                arrT[current] = tPrime
                current = current + 1

            scores = self.test_one_step(arrH, arrR, arrT)

            rankh = rankt = 1
            for i in range(1, totalTriples):
                if scores[0] > scores[i]:
                    if i < corruptedHeadsEnd:
                        rankh = rankh + 1
                    else:
                        rankt = rankt + 1
                    #triple = Triple(arrH[i], arrR[i], arrT[i])
                    #materializedTriplesAll.add(triple)
                    #if arrR[i] in self.undesired:
                    #    materializedTriplesUndesired.add(triple)

            for m in metrics:
                if m.endswith("All") or \
                        (m.endswith("Undesired") and (t.h in self.undesired_ents or t.t in self.undesired_ents or t.r in self.undesired_rels)) or \
                        (m.endswith("Desired") and t.h not in self.undesired_ents and t.t not in self.undesired_ents and t.r not in self.undesired_rels):
                    metrics[m].update(rankh, rankt)

            if pending % 1000 == 0:
                print("Pending:",pending,"Mean rank:",metrics["mr_All"].get(),"; MRR:",metrics["mrr_All"].get(),"; P@1:",
                      metrics["pAt1_All"].get(),"; P@3:",metrics["pAt3_All"].get(),"; P@10:",metrics["pAt10_All"].get())

        totalCorruptedTriples = totalCorruptedTriples / len(testList)
        print("Avg. corrupted triples:",totalCorruptedTriples)
        for option in ["Desired", "Undesired", "All"]:
            print("Metrics:",option,"; Mean rank:",metrics["mr_"+option].get(),"; MRR:",metrics["mrr_"+option].get(),"; P@1:",
                  metrics["pAt1_"+option].get(),"; P@3:",metrics["pAt3_"+option].get(),"; P@10:",metrics["pAt10_"+option].get(),
                  "; Total triples: ",metrics["mr_"+option].total)
        print("Resorted to LCWA:", resortedToLCWA)


class Metric():
    def __init__(self):
        self.valueh = 0
        self.valuet = 0
        self.total = 0

    def update(self, rankh, rankt):
        self.total = self.total + 1

    def get(self):
        ret = 0
        if self.total > 0:
            ret = (self.valueh + self.valuet) / (2 * self.total)
        return ret

class PrecisionAtX(Metric):
    def __init__(self, at):
        super().__init__()
        self.at = at

    def update(self, rankh, rankt):
        super().update(rankh, rankt)
        if rankh <= self.at:
            self.valueh = self.valueh + 1
        if rankt <= self.at:
            self.valuet = self.valuet + 1

class MeanRank(Metric):
    def update(self, rankh, rankt):
        super().update(rankh, rankt)
        self.valueh = self.valueh + rankh
        self.valuet = self.valuet + rankt

class MeanReciprocalRank(Metric):
    def update(self, rankh, rankt):
        super().update(rankh, rankt)
        self.valueh = self.valueh + 1/rankh
        self.valuet = self.valuet + 1/rankt
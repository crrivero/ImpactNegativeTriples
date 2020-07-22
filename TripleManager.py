import numpy as np
import random
import sys
from DataLoader import DataLoader

class TrainDataSampler(object):

    def __init__(self, nbatches, datasampler):
        self.nbatches = nbatches
        self.datasampler = datasampler
        self.batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.batch += 1
        if self.batch > self.nbatches:
            raise StopIteration()
        return self.datasampler()

    def __len__(self):
        return self.nbatches


class TripleManager():
    def __init__(self, path, splits, nbatches=None, neg_ent=None, neg_rel=None, use_bern=False, seed=None, corruptionMode="Global"):
        self.counter = 0
        self.path = path
        # Whether we will use a Bernouilli distribution to determine whether to corruptt head or tail
        self.use_bern = use_bern

        loaders = []
        self.tripleList = []
        headEntities, tailEntities = set(), set()
        headDict, tailDict, dom, ran = {}, {}, {}, {}

        for s in splits:
            loader = DataLoader(path, s)
            self.entityTotal = loader.entityTotal
            self.relationTotal = loader.relationTotal

            loaders = loaders + [loader]
            self.tripleList = self.tripleList + loader.getTriples()
            headEntities = headEntities.union(loader.getHeadEntities())
            tailEntities = tailEntities.union(loader.getTailEntities())

            for r in loader.relations:
                if r in loader.getHeadDict():
                    if r not in headDict:
                        headDict[r] = {}
                    for t in loader.getHeadDict()[r]:
                        if t not in headDict[r]:
                            headDict[r][t] = set()
                        headDict[r][t] = headDict[r][t].union(loader.getHeadDict()[r][t])

                if r in loader.getTailDict():
                    if r not in tailDict:
                        tailDict[r] = {}
                    for h in loader.getTailDict()[r]:
                        if h not in tailDict[r]:
                            tailDict[r][h] = set()
                        tailDict[r][h] = tailDict[r][h].union(loader.getTailDict()[r][h])

                if r in loader.getDomain():
                    if r not in dom:
                        dom[r] = set()
                    dom[r] = dom[r].union(loader.getDomain()[r])

                if r in loader.getRange():
                    if r not in ran:
                        ran[r] = set()
                    ran[r] = ran[r].union(loader.getRange()[r])

        self.nbatches = nbatches
        self.negative_ent = neg_ent
        self.negative_rel = neg_rel

        if self.use_bern:
            # tph: the average number of tail entities per head entity
            # hpt: the average number of head entities per tail entity
            tph, hpt = {}, {}
            relations = set()
            for r in tailDict:
                tph[r] = 0
                for h in tailDict[r]:
                    tph[r] += len(tailDict[r][h])
                tph[r] = tph[r]/len(tailDict[r].keys())
                relations.add(r)
            for r in headDict:
                hpt[r] = 0
                for t in headDict[r]:
                    hpt[r] += len(headDict[r][t])
                hpt[r] = hpt[r]/len(headDict[r].keys())
                relations.add(r)
            self.headProb = {}
            for r in relations:
                self.headProb[r] = tph[r]/(tph[r]+hpt[r])
                #self.tailProb[r] = hpt[r]/(tph[r]+hpt[r])

        if seed is not None:
            random.seed(seed)

        # The entity set must be the same for all
        self.entitySet = set(range(loaders[0].entityTotal))
        self.tripleTotal = len(self.tripleList)
        if self.nbatches is not None:
            self.batchSize = self.tripleTotal // self.nbatches

        self.headCorruptedDict = {}
        self.tailCorruptedDict = {}
        self.corruptionMode = corruptionMode

        self.headCorruptedEntities = {'entities':list(self.entitySet - headEntities), 'counter':0}
        self.tailCorruptedEntities = {'entities':list(self.entitySet - tailEntities), 'counter':0}
        self.resortedToLCWA = 0

        # Nothing to do when using global
        if self.corruptionMode != "Global":
            for r in headDict:
                self.headCorruptedDict[r] = {}

                headEntities = set()
                if self.corruptionMode == "LCWA":
                    headEntities = self.entitySet
                elif self.corruptionMode == "TCLCWA":
                    headEntities = dom[r]
                elif self.corruptionMode == "NLCWA" or self.corruptionMode == "GNLCWA":
                    headEntities = dom[r]
                    # Compatible relations are always the same.
                    for ri in loaders[0].domDomCompatible[r]:
                        headEntities = headEntities.union(ran[ri])
                        #headEntities = headEntities.union(dom[ri])
                    for rj in loaders[0].domRanCompatible[r]:
                        headEntities = headEntities.union(dom[rj])
                        #headEntities = headEntities.union(ran[rj])

                for t in headDict[r]:
                    corruptedHeads = headEntities - headDict[r][t]
                    if len(corruptedHeads) == 0:
                        if self.corruptionMode == "LCWA":
                            print("Corrupted heads were empty using LCWA")
                            sys.exit(-1)
                        else:
                            self.resortedToLCWA = self.resortedToLCWA + 1
                            corruptedHeads = self.entitySet - headDict[r][t]
                    self.headCorruptedDict[r][t] = {'entities':list(corruptedHeads), 'counter':0}

            for r in tailDict:
                self.tailCorruptedDict[r] = {}

                tailEntities = set()
                if self.corruptionMode == "LCWA":
                    tailEntities = self.entitySet
                elif self.corruptionMode == "TCLCWA":
                    tailEntities = ran[r]
                elif self.corruptionMode == "NLCWA" or self.corruptionMode == "GNLCWA":
                    tailEntities = ran[r]
                    for ri in loaders[0].ranRanCompatible[r]:
                        tailEntities = tailEntities.union(dom[ri])
                        #tailEntities = tailEntities.union(ran[ri])
                    for rj in loaders[0].ranDomCompatible[r]:
                        tailEntities = tailEntities.union(ran[rj])
                        #tailEntities = tailEntities.union(dom[rj])

                for h in tailDict[r]:
                    corruptedTails = tailEntities - tailDict[r][h]
                    if len(corruptedTails) == 0:
                        if self.corruptionMode == "LCWA":
                            print("Corrupted tails were empty using LCWA")
                            sys.exit(-1)
                        else:
                            self.resortedToLCWA = self.resortedToLCWA + 1
                            corruptedTails = self.entitySet - tailDict[r][h]
                    self.tailCorruptedDict[r][h] = {'entities':list(corruptedTails), 'counter':0}

    def corrupt_head(self, h, r, t):
        useGlobal = random.random() < 0.25
        if self.corruptionMode == "Global" or (self.corruptionMode == "GNLCWA" and useGlobal):
            hPrime = self.next_corrupted(self.headCorruptedEntities)
        else:
            hPrime = self.next_corrupted(self.headCorruptedDict[r][t])
        return hPrime

    def corrupt_tail(self, h, r, t):
        useGlobal = random.random() < 0.25
        if self.corruptionMode == "Global" or (self.corruptionMode == "GNLCWA" and useGlobal):
            tPrime = self.next_corrupted(self.tailCorruptedEntities)
        else:
            tPrime = self.next_corrupted(self.tailCorruptedDict[r][h])
        return tPrime

    def next_corrupted(self, dict):
        # Reinitialize counter.
        if len(dict['entities']) == dict['counter']:
            dict['counter'] = 0
        ret = dict['entities'][dict['counter']]
        dict['counter'] = dict['counter'] + 1
        return ret

    def getBatches(self):
        batch_seq_size = self.batchSize * (1 + self.negative_ent + self.negative_rel)
        batch_h = np.zeros(batch_seq_size, dtype=np.int64)
        batch_t = np.zeros(batch_seq_size, dtype=np.int64)
        batch_r = np.zeros(batch_seq_size, dtype=np.int64)
        batch_y = np.zeros(batch_seq_size, dtype=np.float32)

        for batch in range(self.batchSize):
            randIndex = random.randint(0, self.tripleTotal-1)
            batch_h[batch] = self.tripleList[randIndex].h
            batch_t[batch] = self.tripleList[randIndex].t
            batch_r[batch] = self.tripleList[randIndex].r
            batch_y[batch] = 1
            last = self.batchSize

            for times in range(self.negative_ent):
                ch = self.tripleList[randIndex].h
                ct = self.tripleList[randIndex].t

                if random.random() < self.headProb[self.tripleList[randIndex].r] \
                    if self.use_bern else random.random() < 0.5:
                    ch = self.corrupt_head(ch, self.tripleList[randIndex].r, ct)
                else:
                    ct = self.corrupt_tail(ch, self.tripleList[randIndex].r, ct)

                batch_h[batch + last] = ch
                batch_t[batch + last] = ct
                batch_r[batch + last] = self.tripleList[randIndex].r
                batch_y[batch + last] = -1
                last = last + self.batchSize

        return {
            "batch_h": batch_h,
            "batch_t": batch_t,
            "batch_r": batch_r,
            "batch_y": batch_y,
            "mode": "normal"
        }

    def __next__(self):
        self.counter += 1
        if self.counter > self.nbatches:
            raise StopIteration()
        return self.getBatches()

    def __iter__(self):
        return TrainDataSampler(self.nbatches, self.getBatches)

    def __len__(self):
        return self.nbatches
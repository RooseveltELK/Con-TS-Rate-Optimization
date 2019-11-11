from __future__ import division

import time
import numpy as np


class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError


class BernoulliBandit(Bandit):

    def __init__(self, probas=None):
        assert probas is None or len(probas) == n
        self.ss_rates = [13.5,27,40.5,54,81,108,121.5,135] # Collection of single-stream rate in Mbps
        self.ds_rates = [27,54,81,108,162,216,243,270] # Collection of double-stream rate in Mbps
        self.n = len(self.ss_rates)
        if probas is None: # I need to change here according to certain rate transmission success probability
            np.random.seed(int(time.time()))
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas
        self.throughput = [self.probas[x]*self.ss_rates[x] for x in range(self.n)]
        self.best_throughput = max(self.throughput)

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probas[i]:
            return 1
        else:
            return 0

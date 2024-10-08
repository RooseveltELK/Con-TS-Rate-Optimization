from __future__ import division

import time
import numpy as np
import math

class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError


class BernoulliBandit(Bandit):

    def __init__(self, rate_set, probas):
        assert probas is not None
        self.ss_rates = rate_set # Collection of rate in Mbps
        
        self.n = len(self.ss_rates)
        if probas is None: # I need to change here according to certain rate transmission success probability
            
            self.probas = [0.95,0.90,0.80,0.65,0.45,0.25,0.15,0.10]
        else:
            self.probas = probas
        #self.throughput = [self.probas[x]*self.ss_rates[x] for x in range(self.n)]
        self.throughput = [self.probas[x]*math.log(self.ss_rates[x]) for x in range(self.n)] # Apply utility function
        self.best_throughput = max(self.throughput)
        self.best_arm = max(range(self.n), key=lambda x: self.throughput[x])

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probas[i]:
            return 1
        else:
            return 0

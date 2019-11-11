from __future__ import division

import numpy as np
import time
from scipy.stats import beta

from bandits import BernoulliBandit


class Solver(object): # In Python 3, we can use Solver instead of Solver(object) in Python 2.
    def __init__(self, bandit):
        """
        bandit (Bandit): the target bandit to solve.
        """
        assert isinstance(bandit, BernoulliBandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.n
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.

    def update_regret(self, i):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_throughput - self.bandit.throughput[i] # Regret is the loss of throughput due to suboptimal rate selection
        self.regrets.append(self.regret)

    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)

class EpsilonGreedy(Solver):
    def __init__(self, bandit, eps, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        super(EpsilonGreedy, self).__init__(bandit)

        assert 0. <= eps <= 1.0
        self.eps = eps

        self.estimates = [init_proba] * self.bandit.n  # Optimistic initialization

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        if np.random.random() < self.eps:
            # Let's do random exploration!
            i = np.random.randint(0, self.bandit.n)
        else:
            # Pick the best one.
            throughput = self.estimates*self.bandit.ss_rates
            i = max(range(self.bandit.n), key=lambda x: self.estimates[x])

        r = self.bandit.generate_reward(i)
        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i


#class UCB1(Solver):
#    def __init__(self, bandit, init_proba=1.0):
#        super(UCB1, self).__init__(bandit)
#        self.t = 0
#        self.estimates = [init_proba] * self.bandit.n

#    @property
#    def estimated_probas(self):
#        return self.estimates

#    def run_one_step(self):
#        self.t += 1

#        # Pick the best one with consideration of upper confidence bounds.
#        i = max(range(self.bandit.n), key=lambda x: self.estimates[x] + np.sqrt(
#            2 * np.log(self.t) / (1 + self.counts[x])))
#        r = self.bandit.generate_reward(i)

#        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

#        return i


#class BayesianUCB(Solver):
#    """Assuming Beta prior."""

#    def __init__(self, bandit, c=3, init_a=1, init_b=1):
#        """
#        c (float): how many standard dev to consider as upper confidence bound.
#        init_a (int): initial value of a in Beta(a, b).
#        init_b (int): initial value of b in Beta(a, b).
#        """
#        super(BayesianUCB, self).__init__(bandit)
#        self.c = c
#        self._as = [init_a] * self.bandit.n
#        self._bs = [init_b] * self.bandit.n

#    @property
#    def estimated_probas(self):
#        return [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

#    def run_one_step(self):
#        # Pick the best one with consideration of upper confidence bounds.
#        i = max(
#            range(self.bandit.n),
#            key=lambda x: self._as[x] / float(self._as[x] + self._bs[x]) + beta.std(
#                self._as[x], self._bs[x]) * self.c
#        )
#        r = self.bandit.generate_reward(i)

#        # Update Gaussian posterior
#        self._as[i] += r
#        self._bs[i] += (1 - r)

#        return i


class ThompsonSampling(Solver):
    def __init__(self, bandit, init_a=1, init_b=1):
        """
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(ThompsonSampling, self).__init__(bandit) # In Python 3, this sentence can be simplified as super().__init__(bandit)

        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit.n)]
        throughput = [samples[x]*self.bandit.ss_rates[x] for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: throughput[x])
        r = self.bandit.generate_reward(i)

        self._as[i] += r
        self._bs[i] += (1 - r)

        return i

class ConTS(Solver): # I will first assort rate from slow to fast
    def __init__(self, bandit, init_a=1, init_b=1):
        """
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(ConTS, self).__init__(bandit) # In Python 3, this sentence can be simplified as super().__init__(bandit)

        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        for x in range(self.bandit.n): # Sample Beta distribution with ordering to speed up
            value = np.random.beta(self._as[x], self._bs[x])
            while(x > 0 and value > samples[x-1]):
                value = np.random.beta(self._as[x], self._bs[x])
            samples[x] = value
        throughput = [samples[x]*self.bandit.ss_rates[x] for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: throughput[x])
        r = self.bandit.generate_reward(i)

        self._as[i] += r
        self._bs[i] += (1 - r)

        return i
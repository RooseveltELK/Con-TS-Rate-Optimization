from __future__ import division

import numpy as np
import time
from scipy.stats import beta, entropy

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
        if self.bandit.best_arm != i:
            a = 1
        else:
            a = 0
        self.regret += a # Regret is the loss of throughput due to suboptimal rate selection
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

    def run(self,num_steps,rate_set,probas1,probas2,probas3):
        assert self.bandit is not None
        for step in range(num_steps):

            if step < num_steps/3:
                self.bandit = BernoulliBandit(rate_set, probas1)
            elif step > 2*num_steps/3:
                self.bandit = BernoulliBandit(rate_set, probas3)
            else:
                self.bandit = BernoulliBandit(rate_set, probas2)

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
        self._as = [1] * self.bandit.n
        self._bs = [0] * self.bandit.n
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
            throughput = [self.estimates[x]*self.bandit.ss_rates[x] for x in range(self.bandit.n)]
            i = max(range(self.bandit.n), key=lambda x: throughput[x])

        r = self.bandit.generate_reward(i)
        self._as[i] += r
        self._bs[i] += (1 - r)

        self.estimates[i] = (self._as[i]) / (self._as[i]+self._bs[i])

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

class GORS(Solver):
    def __init__(self, bandit, init_a=0, init_b=0):
         super(GORS, self).__init__(bandit)

         self._as = [init_a] * self.bandit.n
         self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
         return [(self._as[i]) / float(self._as[i]+self._bs[i]) for i in range(self.bandit.n)]


    def run_one_step(self):
        mu_k = [self.bandit.ss_rates[i]*self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit.n)]
        #print('throughput',mu_k)
        # Pick the best one with consideration of upper confidence bounds.
        j = max(range(self.bandit.n), key=lambda x: mu_k[x]) # j is L(n)
        l_j = self.counts[j]
        #print('l_j',l_j)
        q = [0] * self.bandit.n
        for l in range(self.bandit.n):
            for k in range(1,int(self.bandit.ss_rates[l])):
                #print('part 1',float(self._as[l] + self._bs[l]))
                #print('part 2',mu_k[l]/self.bandit.ss_rates[l])
                #print('part 3',k/self.bandit.ss_rates[l])
                #print('part 4',entropy([mu_k[l]/self.bandit.ss_rates[l],1-mu_k[l]/self.bandit.ss_rates[l]],[k/self.bandit.ss_rates[l],1-k/self.bandit.ss_rates[l]]))
                if float(self._as[l] + self._bs[l])*entropy([mu_k[l]/self.bandit.ss_rates[l],1-mu_k[l]/self.bandit.ss_rates[l]],[k/self.bandit.ss_rates[l],1-k/self.bandit.ss_rates[l]]) <= np.log(l_j) + max(2*np.log(np.log(l_j)),0):
                    q[l] = k
                    
        #print(q)
        if (l_j-1)%3 ==0:
            i = j
        elif j == 0:
            i = max(range(j,j+2), key=lambda x: q[x])
        elif j == (self.bandit.n -1):
            i = max(range(j-1,j+1), key=lambda x: q[x])
        else:
            i = max(range(j-1,j+2), key=lambda x: q[x])

        r = self.bandit.generate_reward(i)

       
        # Update Beta posterior
        self._as[i] += r
        self._bs[i] += (1 - r)

        return i

    def run(self, num_steps):
        assert self.bandit is not None
        for x in range(self.bandit.n): # First run each rate once
            self.counts[x] = 1
            self._as[x] = self.bandit.generate_reward(x)
            self._bs[x] = (1 - self._as[x])
        #print('I have run the initial reward')
        #print('success',self._as)
        #print('failure',self._bs)
        for _ in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)

    def run(self, num_steps,rate_set,probas1,probas2,probas3):
        assert self.bandit is not None
        for x in range(self.bandit.n): # First run each rate once
            self.counts[x] = 1
            self._as[x] = self.bandit.generate_reward(x)
            self._bs[x] = (1 - self._as[x])
        #print('I have run the initial reward')
        #print('success',self._as)
        #print('failure',self._bs)
        for step in range(num_steps):

            if step < num_steps/3:
                self.bandit = BernoulliBandit(rate_set, probas1)
            elif step > 2*num_steps/3:
                self.bandit = BernoulliBandit(rate_set, probas3)
            else:
                self.bandit = BernoulliBandit(rate_set, probas2)

            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)   

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
        return [(self._as[i]) / float(self._as[i]+self._bs[i]) for i in range(self.bandit.n)]

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
        return [(self._as[i]) / float(self._as[i]+self._bs[i]) for i in range(self.bandit.n)]

    def csample(self,a,b,_as,_bs): 
        """
        Warning here: I must clare 'self' here. If not, when I use this function there will be an error:
        'takes 4 positional arguments but 5 were given'.
        """
        # a,b are constrained range of sampling value, _as and _bs are parameters of beta distribution
        Z = beta.ppf(np.random.uniform(beta.cdf(a,_as,_bs),beta.cdf(b,_as,_bs)),_as,_bs) # ppf is inverse of cdf
        return Z

    def run_one_step(self):
        samples = [0] * self.bandit.n

        for x in range(self.bandit.n): # Sample Beta distribution with ordering to speed up
            # print(0,samples[x-1],self._as[x],self._bs[x])
            if x == 0:
                samples[x] = self.csample(0,1,self._as[x],self._bs[x])
            else:
                samples[x] = self.csample(0,samples[x-1],self._as[x],self._bs[x])

        # print('sampled probabilities:',samples)
        throughput = [samples[x]*self.bandit.ss_rates[x] for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: throughput[x])
        r = self.bandit.generate_reward(i)

        self._as[i] += r
        self._bs[i] += (1 - r)
        return i
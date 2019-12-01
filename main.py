import matplotlib  # noqa
matplotlib.use('Agg')  # noqa

import matplotlib.pyplot as plt
import numpy as np
from bandits import BernoulliBandit
from solvers import Solver, EpsilonGreedy, ThompsonSampling, ConTS, GORS # UCB1, BayesianUCB


def plot_results(solvers, solver_names, figname):
    """
    Plot the results by multi-armed bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit

    fig = plt.figure(figsize=(14, 4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Sub.fig. 1: Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 2: Probabilities estimated by solvers.
    sorted_indices = sorted(range(b.n), key=lambda x: b.probas[x])
    ax2.plot(range(b.n), [b.probas[x] for x in sorted_indices], 'k--', markersize=12)
    for s in solvers:
        ax2.plot(range(b.n), [s.estimates[x] for x in sorted_indices], 'x', markeredgewidth=2)
    ax2.set_xlabel('Rates sorted by ' + r'$\theta$')
    ax2.set_ylabel('Estimated')
    ax2.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 3: Action counts
    x = np.arange(b.n)
    total_width = 0.8
    width = total_width / b.n
    x = x - (total_width - width) / 4
    for i, s in enumerate(solvers):
        ax3.bar(x + i*width, np.array(s.counts) / float(len(s.regrets)), width=width)
        #ax3.plot(range(b.n), np.array(s.counts) / float(len(s.regrets)), ls='steps', lw=2)
    ax3.set_xlabel('Rate index')
    ax3.set_ylabel('Fractions of rates been chosen')
    ax3.grid('k', ls='--', alpha=0.3)

    plt.savefig(figname)


def experiment(N,K,rate_set,probas):
    """
    Run a small experiment on solving a Bernoulli bandit with K slot machines,
    each with a randomly initialized reward probability.

    Args:
        K (int): number of Monte Carlo repeat time.
        N (int): numbefr of time steps to try.
        rate_set: Set of arms
        probas: Set of probabilities of each arm 
    """

    b = BernoulliBandit(rate_set, probas)
    print ("Randomly generated Bernoulli bandit has reward probabilities:\n", b.probas)
    print ("The best machine has index: {} and throughput: {}".format(
        b.best_arm, b.best_throughput))

    test_solvers = [
        EpsilonGreedy(b, 0.01),
        # UCB1(b),
        # BayesianUCB(b, 3, 1, 1),
        ThompsonSampling(b, 1, 1),
        ConTS(b,1,1),
        GORS(b)
    ]
    names = [
        # 'Full-exploitation',
        # 'Full-exploration',
        r'$\epsilon$' + '-Greedy',
        # 'UCB1',
        # 'Bayesian UCB',
        'Thompson Sampling',
        'Constrained TS',
        'G-ORS'
    ]

    # Do Monte Carlo K times
    regrets_e = [[] * N for i in range(K)]
    regrets_t = [[] * N for i in range(K)]
    regrets_c = [[] * N for i in range(K)]
    regrets_g = [[] * N for i in range(K)]
    estimated_probas_e = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    estimated_probas_t = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    estimated_probas_c = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    estimated_probas_g = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    counts_e = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    counts_t = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    counts_c = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    counts_g = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]

    for i in range(K):
        #for s in test_solvers:
        #    s.run(N)
        s1 = EpsilonGreedy(b, 0.01)
        s1.run(N)
        regrets_e[i] = s1.regrets
        estimated_probas_e[i] = s1.estimated_probas
        counts_e[i] = s1.counts
        s2 = ThompsonSampling(b, 1, 1)
        s2.run(N)
        regrets_t[i] = s2.regrets
        estimated_probas_t[i] = s2.estimated_probas
        counts_t[i] = s2.counts
        s3 = ConTS(b,1,1)
        s3.run(N)
        regrets_c[i] = s3.regrets
        estimated_probas_c[i] = s3.estimated_probas
        counts_c[i] = s3.counts
        s4 = GORS(b)
        s4.run(N)
        regrets_g[i] = s4.regrets
        estimated_probas_g[i] = s4.estimated_probas
        counts_g[i] = s4.counts


    s1.regrets = np.divide(list(map(sum,zip(*regrets_e))),K)
    s2.regrets = np.divide(list(map(sum,zip(*regrets_t))),K)
    s3.regrets = np.divide(list(map(sum,zip(*regrets_c))),K)
    s4.regrets = np.divide(list(map(sum,zip(*regrets_g))),K)
    s1.estimates = np.divide(list(map(sum,zip(*estimated_probas_e))),K)
    s2.estimates = np.divide(list(map(sum,zip(*estimated_probas_t))),K)
    s3.estimates = np.divide(list(map(sum,zip(*estimated_probas_c))),K)
    s4.estimates = np.divide(list(map(sum,zip(*estimated_probas_g))),K)
    s1.counts = np.divide(list(map(sum,zip(*counts_e))),K)
    s2.counts = np.divide(list(map(sum,zip(*counts_t))),K)
    s3.counts = np.divide(list(map(sum,zip(*counts_c))),K)
    s4.counts = np.divide(list(map(sum,zip(*counts_g))),K)
    test_solvers_mod = [
        s1,
        s2,
        s3,
        s4
        ]
    plot_results(test_solvers_mod, names, "results_N{}_K{}.png".format(N,K))

def experiment2(N,K,rate_set,probas):
    """
    Run a small experiment on solving a Bernoulli bandit with K slot machines,
    each with a randomly initialized reward probability.

    Args:
        K (int): number of Monte Carlo repeat time.
        N (int): numbefr of time steps to try.
        rate_set: Set of arms
        probas: Set of probabilities of each arm 
    """

    b = BernoulliBandit(rate_set, probas)
    print ("Randomly generated Bernoulli bandit has reward probabilities:\n", b.probas)
    print ("The best machine has index: {} and throughput: {}".format(
        b.best_arm, b.best_throughput))

    test_solvers = [
        EpsilonGreedy(b, 0.01),
        # UCB1(b),
        # BayesianUCB(b, 3, 1, 1),
        ThompsonSampling(b, 1, 1),
        #ConTS(b,1,1),
        GORS(b)
    ]
    names = [
        # 'Full-exploitation',
        # 'Full-exploration',
        r'$\epsilon$' + '-Greedy',
        # 'UCB1',
        # 'Bayesian UCB',
        'Thompson Sampling',
        #'Constrained TS',
        'G-ORS'
    ]

    # Do Monte Carlo K times
    regrets_e = [[] * N for i in range(K)]
    regrets_t = [[] * N for i in range(K)]
    #regrets_c = [[] * N for i in range(K)]
    regrets_g = [[] * N for i in range(K)]
    estimated_probas_e = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    estimated_probas_t = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    #estimated_probas_c = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    estimated_probas_g = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    counts_e = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    counts_t = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    #counts_c = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    counts_g = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]

    for i in range(K):
        #for s in test_solvers:
        #    s.run(N)
        s1 = EpsilonGreedy(b, 0.01)
        s1.run(N)
        regrets_e[i] = s1.regrets
        estimated_probas_e[i] = s1.estimated_probas
        counts_e[i] = s1.counts
        s2 = ThompsonSampling(b, 1, 1)
        s2.run(N)
        regrets_t[i] = s2.regrets
        estimated_probas_t[i] = s2.estimated_probas
        counts_t[i] = s2.counts
        #s3 = ConTS(b,1,1)
        #s3.run(N)
        #regrets_c[i] = s3.regrets
        #estimated_probas_c[i] = s3.estimated_probas
        #counts_c[i] = s3.counts
        s4 = GORS(b)
        s4.run(N)
        regrets_g[i] = s4.regrets
        estimated_probas_g[i] = s4.estimated_probas
        counts_g[i] = s4.counts


    s1.regrets = np.divide(list(map(sum,zip(*regrets_e))),K)
    s2.regrets = np.divide(list(map(sum,zip(*regrets_t))),K)
    #s3.regrets = np.divide(list(map(sum,zip(*regrets_c))),K)
    s4.regrets = np.divide(list(map(sum,zip(*regrets_g))),K)
    s1.estimates = np.divide(list(map(sum,zip(*estimated_probas_e))),K)
    s2.estimates = np.divide(list(map(sum,zip(*estimated_probas_t))),K)
    #s3.estimates = np.divide(list(map(sum,zip(*estimated_probas_c))),K)
    s4.estimates = np.divide(list(map(sum,zip(*estimated_probas_g))),K)
    s1.counts = np.divide(list(map(sum,zip(*counts_e))),K)
    s2.counts = np.divide(list(map(sum,zip(*counts_t))),K)
    #s3.counts = np.divide(list(map(sum,zip(*counts_c))),K)
    s4.counts = np.divide(list(map(sum,zip(*counts_g))),K)
    test_solvers_mod = [
        s1,
        s2,
        #s3,
        s4
        ]
    plot_results(test_solvers_mod, names, "results_N{}_K{}.png".format(N,K))

def experiment3(N,K,rate_set,probas1,probas2,probas3):
    """
    Run a small experiment on solving a Bernoulli bandit with K slot machines,
    each with a randomly initialized reward probability.

    Args:
        K (int): number of Monte Carlo repeat time.
        N (int): numbefr of time steps to try.
        rate_set: Set of arms
        probas1,probas2,probas3: Different sets of probabilities of each arm 
    """

    b = BernoulliBandit(rate_set, probas1)
    #print ("Randomly generated Bernoulli bandit has reward probabilities:\n", b.probas)
    #print ("The best machine has index: {} and throughput: {}".format(
    #    b.best_arm, b.best_throughput))

    test_solvers = [
        EpsilonGreedy(b, 0.01),
        # UCB1(b),
        # BayesianUCB(b, 3, 1, 1),
        ThompsonSampling(b, 1, 1),
        #ConTS(b,1,1),
        GORS(b)
    ]
    names = [
        # 'Full-exploitation',
        # 'Full-exploration',
        r'$\epsilon$' + '-Greedy',
        # 'UCB1',
        # 'Bayesian UCB',
        'Thompson Sampling',
        #'Constrained TS',
        'G-ORS'
    ]

    # Do Monte Carlo K times
    regrets_e = [[] * N for i in range(K)]
    regrets_t = [[] * N for i in range(K)]
    #regrets_c = [[] * N for i in range(K)]
    regrets_g = [[] * N for i in range(K)]
    estimated_probas_e = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    estimated_probas_t = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    #estimated_probas_c = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    estimated_probas_g = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    counts_e = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    counts_t = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    #counts_c = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]
    counts_g = [[] * len(test_solvers[0].bandit.ss_rates) for i in range(K)]

    for i in range(K):
        #for s in test_solvers:
        #    s.run(N)
        s1 = EpsilonGreedy(b, 0.01)
        s1.run(N,rate_set,probas1,probas2,probas3)
        regrets_e[i] = s1.regrets
        estimated_probas_e[i] = s1.estimated_probas
        counts_e[i] = s1.counts
        s2 = ThompsonSampling(b, 1, 1)
        s2.run(N,rate_set,probas1,probas2,probas3)
        regrets_t[i] = s2.regrets
        estimated_probas_t[i] = s2.estimated_probas
        counts_t[i] = s2.counts
        #s3 = ConTS(b,1,1)
        #s3.run(N)
        #regrets_c[i] = s3.regrets
        #estimated_probas_c[i] = s3.estimated_probas
        #counts_c[i] = s3.counts
        s4 = GORS(b)
        s4.run(N,rate_set,probas1,probas2,probas3)
        regrets_g[i] = s4.regrets
        estimated_probas_g[i] = s4.estimated_probas
        counts_g[i] = s4.counts


    s1.regrets = np.divide(list(map(sum,zip(*regrets_e))),K)
    s2.regrets = np.divide(list(map(sum,zip(*regrets_t))),K)
    #s3.regrets = np.divide(list(map(sum,zip(*regrets_c))),K)
    s4.regrets = np.divide(list(map(sum,zip(*regrets_g))),K)
    s1.estimates = np.divide(list(map(sum,zip(*estimated_probas_e))),K)
    s2.estimates = np.divide(list(map(sum,zip(*estimated_probas_t))),K)
    #s3.estimates = np.divide(list(map(sum,zip(*estimated_probas_c))),K)
    s4.estimates = np.divide(list(map(sum,zip(*estimated_probas_g))),K)
    s1.counts = np.divide(list(map(sum,zip(*counts_e))),K)
    s2.counts = np.divide(list(map(sum,zip(*counts_t))),K)
    #s3.counts = np.divide(list(map(sum,zip(*counts_c))),K)
    s4.counts = np.divide(list(map(sum,zip(*counts_g))),K)
    test_solvers_mod = [
        s1,
        s2,
        #s3,
        s4
        ]
    plot_results(test_solvers_mod, names, "results_N{}_K{}_time-varing.png".format(N,K))

if __name__ == '__main__':
    # Single-stream mode
    #rate_set = [13.5,27,40.5,54,81,108,121.5,135]
    #probas = [0.999007693923863,0.984935319629647,0.964540932372162,0.912156133797521,0.880122515494731,0.809960844711017,0.803708566959608,0.798681977496734]
    #experiment(10000,50,rate_set,probas)
    
    # Both single-stream and double-stream modes
    #rate_set = [13.5,27,40.5,54,81,108,121.5,135,27,54,81,108,162,216,243,270]
    #probas = [0.999007693923863,0.984935319629647,0.964540932372162,0.912156133797521,0.880122515494731,0.809960844711017,0.803708566959608,0.798681977496734,0.998016372519075,0.970097583853955,0.930339210221360,0.832028812424441,0.774615642280773,0.656036569964984,0.645947460604267,0.637892901178094]
    #experiment2(10000,50,rate_set,probas)

    # In time-varing channel with 3 states change
    rate_set = [13.5,27,40.5,54,81,108,121.5,135,27,54,81,108,162,216,243,270]
    probas1 = [0.999007693923863,0.984935319629647,0.964540932372162,0.912156133797521,0.880122515494731,0.809960844711017,0.803708566959608,0.798681977496734,0.998016372519075,0.970097583853955,0.930339210221360,0.832028812424441,0.774615642280773,0.656036569964984,0.645947460604267,0.637892901178094]
    probas2 = [0.9,0.8,0.7,0.55,0.45,0.35,0.2,0.1,0.81,0.64,0.49,0.3025,0.2025,0.1225,0.04,0.01]
    probas3 = [0.95,0.9,0.8,0.65,0.45,0.25,0.15,0.1,0.9025,0.81,0.64,0.4225,0.2025,0.0625,0.0225,0.01]
    experiment3(30000,50,rate_set,probas1,probas2,probas3)
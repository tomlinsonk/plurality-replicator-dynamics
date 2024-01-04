import argparse
from functools import partial
import itertools
from multiprocessing import Pool
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm
from collections import deque


def plurality_votes(cand_pos, voter_dsn=None):
    """
    Compute the vote shares of candidates in a collection of elections,
    optionally with the given voter distribution (default: uniform voters).

    @cand_pos an (n, k) numpy array (n = #elections, k = #candidates) of
              candidate positions
    @voter_dsn a scipy.stats distribution; if None (default), use uniform voters

    @returns a length (n, k) numpy array of sorted cands, and a length (n, k)
             numpy array votes where votes[i,j] is the fraction of votes for
             candidate i, j in the sorted cands array
    """

    n, k = cand_pos.shape
    sorted_cands = np.sort(cand_pos, axis=1)
    regions = np.column_stack(
        (
            np.zeros((n, 1)),
            (sorted_cands[:, 1:] + sorted_cands[:, :-1]) / 2,
            np.ones((n, 1)),
        )
    )
    cdfs = regions if voter_dsn is None else voter_dsn.cdf(regions)
    votes = np.diff(cdfs)

    return sorted_cands, votes


def plurality(cand_pos, voter_dsn=None):
    """
    Compute the positions of the plurality winners in a collection of elections,
    optionally with the given voter distribution (default: uniform voters).

    @cand_pos an (n, k) numpy array (n = #elections, k = #candidates) of
              candidate positions
    @voter_dsn a scipy.stats distribution; if None (default), use uniform voters

    @returns a length n numpy array of the winner positions in each election
    """

    sorted_cands, votes = plurality_votes(cand_pos, voter_dsn)
    winner_idxs = np.argmax(votes, axis=1, keepdims=True)
    winners = np.take_along_axis(sorted_cands, winner_idxs, axis=1).flatten()

    return winners


def replicator(
    k,
    n,
    gens,
    symmetry=False,
    initial_dsn=None,
    voter_dsn=None,
    memory=1,
    uniform_eps=0,
    perturb_stdev=0,
    h=1,
    min=0,
    max=1,
    n_bins=100,
    rng=None
):
    """
    Run the replicator dynamics with k candidates per election, n elections per
    generation, and the given number of generations. Return per-generation
    histograms.
    Optionally, add per-point symmetry, use a custom initial distribution
    (default: uniform), a custom voter distribution (default: uniform), more
    generations of memory, epsilon-uniform noise, or normal perturbation noise.

    @k the number of candidates per election
    @n the number of elections per generation
    @gens the number of generations to run
    @symmetry if True, mirror each candidate across 1/2 w.p. 1/2
    @initial_dsn a scipy.stats distribution for the initial candidate
                 distribution F_0; if None (default), start uniform
    @voter_dsn a scipy.stats distribution; if None (default), use uniform voters
    @memory the number of generations back to sample winnners from (default 1)
    @uniform_noise_epsilon the fraction of candidate which are uniform rather
                           than winner samplers (default 0)
    @perturbation_noise_stdev the amount by which to perturb each point with
                              Gaussian noise (default 0)
    @h the number of top candidates to sample winners from (default 1)
    @min the lowest possible candidate position (default 0)
    @max the highest possible candidate position (default 1)
    @n_bins the number of bins in the returned histograms (default 100)

    @return the tuple (hists, bin_edges) where hists is a (gens, n_bins) numpy
            array with winner counts in each bin and bin_edges is a length
            nbins+1 numpy array of the bin edges (as in np.histogram)
    """
    if initial_dsn is None:
        initial_dsn = stats.uniform(min, max)

    if rng is None:
        rng = np.random.default_rng()

    # maintain a queue of the top @h candidates in the last @memory generations
    prev_positions = deque(maxlen=memory)
    prev_positions.append(initial_dsn.rvs(n))

    # save initial candidate distribution
    hists = []
    bins = np.linspace(0, 1, n_bins)
    hist, bin_edges = np.histogram(prev_positions[0], bins=bins)
    hists.append(hist)

    for t in range(gens):
        # combine winner positions from all remembered generations
        sample_from = np.concatenate(prev_positions)
        elections = rng.choice(sample_from, (n, k))

        # add perturbation noise
        if perturb_stdev > 0:
            elections += rng.normal(0, perturb_stdev, (n, k))
            elections = elections.clip(min, max)

        # add an epsilon-fraction of uniform candidates
        if uniform_eps > 0:
            uniform_idxs = rng.binomial(1, uniform_eps, (n, k)) == 1
            elections[uniform_idxs] = (rng.random((n, k)) * (max - min) + min)[uniform_idxs]

        # mirror candidates if using symmetry
        if symmetry:
            flip_idx = rng.integers(0, 2, (n, k)) == 1
            elections[flip_idx] = (1 - elections)[flip_idx]

        # get winner positions
        winners = plurality(elections, voter_dsn)
        if h == 1:
            prev_positions.append(winners)

        # get top h positions
        elif h > 1:
            sorted_cands, votes = plurality_votes(elections, voter_dsn)
            candidate_options = np.array([])
            top_h_indices = np.argpartition(votes, -h, axis=1)[:, -h:].reshape(h * n, 1)
            top_h_indices = np.hstack(
                ((np.arange(n).repeat(h)).reshape(h * n, 1), top_h_indices)
            )
            top_h_positions = sorted_cands[top_h_indices[:, 0], top_h_indices[:, 1]]
            prev_positions.append(top_h_positions)

        hist, bin_edges = np.histogram(winners, bins=bins)
        hists.append(hist)

    return np.array(hists), bin_edges


def replicator_helper(arg_setting, arg_names, n, gens):
    """
    Helper method for running replicator in parallel. Seeds the RNG according
    to the argument settings for replicator (plus the trial ID).

    @arg_setting a list of argument values for replicator (last item: trial ID)
    @arg_names the names of the arguments for replicator, in the same order as ^
    @n the number of elections per generation
    @gens the number of generations

    @return the tuple (arg_settings, hists, edges), with trial ID stripped from
            arg_settings 
    """

    kwargs = {key: val for key, val in zip(arg_names, arg_setting[:-1])}
    kwargs['n'] = n
    kwargs['gens'] = gens
    kwargs['rng'] = np.random.default_rng(abs(hash(arg_setting)))

    return (arg_setting[:-1],) + replicator(**kwargs)



def run_experiment(name, n, gens, trials, threads, arg_dict):
    """
    Run a replicator experiment in parallel. Saves to results/. Runs every
    combination of argument values from arg_dict. Results are saved as per-
    generation histograms (with counts summed over trials).

    @name the name of the experiment (used for results file)
    @n the number of elections per generation
    @gens the number of generations to run
    @trials the number of replicates to run per arg setting
    @threads the number of threads to use
    @arg_dict a dict whose keys are args to replicator() and whose values are
              lists of argument settings.
    """
    assert 'k' in arg_dict, 'must specify k range in arg_dict'

    arg_names = sorted(arg_dict.keys())
    arg_lists = [list(arg_dict[p]) for p in arg_names]
    helper = partial(replicator_helper, arg_names=arg_names, n=n, gens=gens)

    settings = itertools.product(*(arg_lists + [list(range(trials))]))
    total = trials * np.product([len(p) for p in arg_lists])
    results = dict()

    with Pool(threads) as pool:
        for setting, hists, edges in tqdm(pool.imap_unordered(helper, settings),
                                          total=total):
            if setting not in results:
                results[setting] = hists
            else:
                results[setting] += hists

    os.makedirs('results/', exist_ok=True)
    with open(f'results/{name}.pickle', 'wb') as f:
        pickle.dump((results, n, gens, trials, arg_names, arg_lists, edges), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # small_k_no_noise()

    run_experiment(
        'small-k-eps-range-50-trials', n=100_000, gens=300, trials=50, threads=8,
        arg_dict={
            'k': range(2, 11),
            'uniform_eps': [0, 0.001, 0.01, 0.1]
        }               
    )



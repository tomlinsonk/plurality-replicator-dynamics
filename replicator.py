import argparse
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm
from collections import deque


def plurality(cand_pos, voter_dsn=None):
    """
    Compute the positions of the plurality winners in a collection of elections,
    optionally with the given voter distribution (default: uniform voters).

    @cand_pos an (n, k) numpy array (n = #elections, k = #candidates) of
              candidate positions
    @voter_dsn a scipy.stats distribution; if None (default), use uniform voters

    @returns a length n numpy array of the winner positions in each election
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
    winner_idxs = np.argmax(votes, axis=1, keepdims=True)
    winners = np.take_along_axis(sorted_cands, winner_idxs, axis=1).flatten()

    return winners


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
    n_bins=100,
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
    @n_bins the number of bins in the returned histograms (default 100)

    @return the tuple (hists, bin_edges) where hists is a (gens, n_bins) numpy
            array with winner counts in each bin and bin_edges is a length
            nbins+1 numpy array of the bin edges (as in np.histogram)
    """
    if initial_dsn is None:
        initial_dsn = stats.uniform(0, 1)

    # maintain a queue of the top @h candidates in the last @memory generations
    prev_winners = deque(maxlen=memory)
    prev_winners.append(initial_dsn.rvs(n))

    # save initial candidate distribution
    hists = []
    bins = np.linspace(0, 1, n_bins)
    hist, bin_edges = np.histogram(prev_winners[0], bins=bins)
    hists.append(hist)

    for t in range(gens):
        # combine winner positions from all remembered generations
        sample_from = np.concatenate(prev_winners)
        elections = np.random.choice(sample_from, (n, k))

        # add perturbation noise
        if perturb_stdev > 0:
            elections += np.random.normal(0, perturb_stdev, (n, k))
            elections = elections.clip(0, 1)

        # add an epsilon-fraction of uniform candidates
        if uniform_eps > 0:
            uniform_idxs = np.random.binomial(1, uniform_eps, (n, k)) == 1
            elections[uniform_idxs] = np.random.rand(n, k)[uniform_idxs]

        # mirror candidates if using symmetry
        if symmetry:
            flip_idx = np.random.randint(0, 2, (n, k)) == 1
            elections[flip_idx] = (1 - elections)[flip_idx]

        # get winner positions
        winners = plurality(elections, voter_dsn)
        if h == 1:
            prev_winners.append(winners)

        if h > 1:
            sorted_cands, votes = plurality_votes(elections, voter_dsn)
            candidate_options = np.array([])
            for i in range(0, h):
                i_place_indexes = np.argpartition(votes, -(i + 1), axis=1)[
                    :, -(i + 1)
                ].reshape(n, 1)
                i_place_indexes = np.hstack(
                    (np.arange(n).reshape(n, 1), i_place_indexes)
                )
                i_place_positions = sorted_cands[
                    i_place_indexes[:, 0], i_place_indexes[:, 1]
                ]
                candidate_options = np.append(candidate_options, i_place_positions)
            prev_winners.append(candidate_options)

        hist, bin_edges = np.histogram(winners, bins=bins)
        hists.append(hist)

    return np.array(hists), bin_edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    hists, edges = replicator(
        k=7,
        n=50_000,
        gens=500,
        perturb_stdev=0.001,
        uniform_eps=0,
        memory=1,
        h=2,
    )

    plt.imshow(
        np.log(1 + hists.T), cmap="afmhot_r", aspect="auto", interpolation="nearest"
    )
    plt.show()

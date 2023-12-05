import argparse
import numpy as np
from tqdm import tqdm


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
    regions = np.column_stack((
        np.zeros((n, 1)), 
        (sorted_cands[:, 1:] + sorted_cands[:, :-1]) / 2,
        np.ones((n, 1))
    ))
    cdfs = regions if voter_dsn is None else voter_dsn.cdf(regions)
    votes = np.diff(cdfs)
    winner_idxs = np.argmax(votes, axis=1, keepdims=True)
    winners = np.take_along_axis(sorted_cands, winner_idxs, axis=1).flatten()

    return winners


def replicator(k, n, gens, symmetry=False, initial_dsn=None, voter_dsn=None,
               memory=1, top_h=1, uniform_noise_epsilon=0, 
               perturbation_noise_stdev=0, n_bins=100):
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
    @top_h instead of just sampling winners, sample from the top h plurality
           shares (default 1; i.e., only sample winners)
    @uniform_noise_epsilon the fraction of candidate which are uniform rather
                           than winner samplers (default 0)
    @perturbation_noise_stdev the amount by which to perturb each point with
                              Gaussian noise (default 0)
    @n_bins the number of bins in the returned histograms (default 100)

    @return the tuple (hists, bin_edges) where hists is a (gens, n_bins) numpy 
            array with winner counts in each bin and bin_edges is a length 
            nbins+1 numpy array of the bin edges (as in np.histogram)

    """
    ... #TODO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
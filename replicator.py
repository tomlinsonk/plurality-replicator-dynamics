import argparse
import numpy as np
from tqdm import tqdm


def plurality(cand_pos, voter_dsn=None):
    """
    Compute the positions of the plurality winners in a collection of elections,
    optionally with the given voter distribution (default: uniform voters)
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_small_k_no_noise():
    with open(f'results/small-k-eps-range-50-trials.pickle', 'rb') as f:
        results, n, gens, trials, arg_names, arg_lists, edges = pickle.load(f)

    ks = [2, 3, 4, 5, 6, 7]

    fig, axes = plt.subplots(1, len(ks), figsize=(8, 3))

    for i, k in enumerate(ks):
        axes[i].imshow(np.log(1 + results[k, 0].T), cmap='afmhot_r', aspect='auto', interpolation='nearest')
    plt.show()




if __name__ == '__main__':
    plot_small_k_no_noise()

    # 
    # plt.show()
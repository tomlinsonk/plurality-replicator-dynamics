import os
import signal
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib

# avoid type 3 fonts
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# allow keyboard interrupt to close pyplot
signal.signal(signal.SIGINT, signal.SIG_DFL) 


def plot_small_k_no_noise(name, epsilon=0):
    with open(f'results/{name}.pickle', 'rb') as f:
        results, n, gens, trials, arg_names, arg_lists, edges = pickle.load(f)

    ks = [2, 3, 4, 5, 6, 7, 8]

    fig, axes = plt.subplots(1, len(ks), figsize=(15, 2))

    for i, k in enumerate(ks):
        axes[i].imshow(np.log(1 + results[k, epsilon].T), cmap='afmhot_r', aspect='auto', interpolation='nearest')
        axes[i].set_xticks([0, 100, 200])
        axes[i].set_yticks([0, 49, 99])
        axes[i].set_yticklabels([1, 0.5, 0] if i == 0 else [])
        axes[i].set_xlabel('$t$')

        axes[i].set_title(f'$k = {k}$')

    plt.savefig(f'plots/{name}-eps-{epsilon}.pdf', bbox_inches='tight', dpi=500)
    plt.close()




if __name__ == '__main__':
    os.makedirs('plots/', exist_ok=True)


    plot_small_k_no_noise('small-k-eps-range-50-trials', epsilon=0)
    plot_small_k_no_noise('small-k-eps-range-50-trials', epsilon=0.01)
    plot_small_k_no_noise('small-k-eps-range-1-trial', epsilon=0)
    plot_small_k_no_noise('small-k-eps-range-1-trial', epsilon=0.01)

    # plot_small_k_no_noise('small-k-eps-range-1-trial', epsilon=0)
    # plot_small_k_no_noise('small-k-eps-range-1-trial', epsilon=0.001)
    # plot_small_k_no_noise('small-k-eps-range-1-trial', epsilon=0.1)

    # 
    # plt.show()
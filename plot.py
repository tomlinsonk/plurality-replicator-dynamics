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


def plot_small_k_no_noise(name, epsilon=0, symmetry=False):
    with open(f'results/{name}.pickle', 'rb') as f:
        results, n, gens, trials, arg_names, arg_lists, edges = pickle.load(f)

    ks = [2, 3, 4, 5, 6, 7]

    fig, axes = plt.subplots(1, len(ks), figsize=(15, 2))

    for i, k in enumerate(ks):
        hists = results[k, True, epsilon] if symmetry else results[k, epsilon]
        axes[i].imshow(np.log(1 + hists.T), cmap='afmhot_r', aspect='auto', interpolation='nearest')
        axes[i].set_xticks([0, 100, 200])
        axes[i].set_yticks([0, 49, 99])
        axes[i].set_yticklabels([1, 0.5, 0] if i == 0 else [])
        axes[i].set_xlabel('$t$')

        axes[i].set_title(f'$k = {k}$')

    plt.savefig(f'plots/{name}-eps-{epsilon}.pdf', bbox_inches='tight', dpi=500)
    plt.close()


def plot_cdf_bounds():
    with open('results/small-k-eps-range-symmetry-50-trials.pickle', 'rb') as f:
        results, n, gens, trials, arg_names, arg_lists, edges = pickle.load(f)

    ks = (2, 3, 4)

    idxs = (
        (9, 21, 32, 43),
        (9, 39, 46),
        (9, 39, 46) 
    )

    k2_pred = lambda p0, t: (2 * p0) ** (2 ** t) / 2
    k3_pred = lambda p0, t: p0 * (3/4 + p0 ** 2) ** t
    k4_pred = lambda p0, t: p0 * (1 - 4 * (1/2 - (p0 / 3 + 1/3)) ** 3) ** t

        
        
    preds = (k2_pred, k3_pred, k4_pred)

    fracs = (
        np.cumsum(results[2, True, 0].T, axis=0) / (trials * n),
        np.cumsum(results[3, True, 0].T, axis=0) / (trials * n),
        np.cumsum(results[4, True, 0].T, axis=0) / (trials * n)
    )

    colors = (
        ['#75888a', '#8e50aa', '#99c15f', '#b45948'],
        ['#8e50aa', '#99c15f', '#b45948'],
        ['#8e50aa', '#99c15f', '#b45948']
    )


    label_pos = (
        ((0.11, 0.105), (0.12, 0.225), (0.105, 0.34), (0.05, 0.455)),
        ((0.4, 0.1), (0.5, 0.396), (0.6, 0.474)),
        ((1.1, 0.11), (1.1, 0.375), (1.1, 0.428))
    )

    rotations = (
        (-42, -53, -50, -28),
        (-28, -45, -19),
        (-3, -1, -1)
    )

    labels = (
        'Theorem 2 (exact)',
        'Theorem 3 (bound)',
        'Theorem 4 (bound)'
    )

    max_ts = (6, 20, 40)

    fig, axes = plt.subplots(1, 3, figsize=(12, 2.6))

    for i, k in enumerate(ks):

        for j, idx in enumerate(idxs[i]):
            is_last = j == len(idxs[i]) - 1
            axes[i].plot(np.arange(max_ts[i] + 1), fracs[i][idx, :max_ts[i]+1], '.', c=colors[i][j],
                label='simulation' if is_last else '')
            
            thm_ts = np.linspace(0, max_ts[i] + 1, 200)
            axes[i].plot(thm_ts, [preds[i](edges[idx+1], t) for t in thm_ts], 
                '-', c=colors[i][j], label=labels[i] if is_last else '')
            axes[i].text(*label_pos[i][j], f'$x = {edges[idx+1]:.2f}$', color=colors[i][j], 
                fontsize=8, rotation=rotations[i][j], rotation_mode='anchor')
        axes[i].legend(fontsize=9, loc='best')
        axes[i].set_xlabel('t')
        axes[i].set_title(f'$k = {k}$')
        axes[i].set_ylim(-0.02, 0.5)

    axes[0].set_ylabel(f'$F_{{k, t}}(x)$')

    plt.subplots_adjust(wspace=0.2)
    # plt.show()
    plt.savefig('plots/pred-vs-sim.pdf', bbox_inches='tight')
    plt.close()


def plot_noisy_convergence():
    with open('results/k-2-3-4-many-epsilon-symmetry-50-trials.pickle', 'rb') as f:
        results, n, gens, trials, arg_names, arg_lists, edges = pickle.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.6))

    ks = (2, 3, 4)

    xs = edges[1:]

    k2_pred = lambda x, eps: (1 - 4 * x * eps * (1 - eps) - (1 - 8 * eps * x * (1 - eps)) ** 0.5) / (4 * (1 - eps) ** 2)
    k4_pred = lambda x, eps: eps / (8 * (1/2 - eps * (x / 3 + 1/3) - (1 - eps) * (x/3 + 1/3)) ** 3)
    preds = (
        k2_pred,
        lambda x, eps: 1.5 * eps,
        k4_pred
    )

    x_idxs = (
        (9, 19, 29, 39),
        (9, 19, 29, 39),
        (33, 39, 41, 44)
    )

    epsilons = (
        (0.1, 1/4, 1/3, 1/2),
        (0.1, 1/4, 1/3, 1/2),
        (0.001, 0.00001, 0.0001, 0.00001)
    )


    colors = (
        ('#75888a', '#8e50aa', '#99c15f', '#b45948'),
        ('#75888a', '#8e50aa', '#99c15f', '#b45948'),
        ('#75888a', '#8e50aa', '#99c15f', '#b45948')
    )

    labels = (
        'Theorem 7 (exact)',
        'Theorem 8 (bound)',
        'Theorem 9 (bound)'
    )

    text_pos = (
        ((15.5, 0.00029), (15.5, 0.0085), (15.5, 0.04), (15.5, 0.22)),
        ((30.5, 0.00004), (30.5, 0.0012), (30.5, 0.014), (30.5, 0.08)),
        ((22, 0.00002), (44, 0.00002), (59, 0.000014), (60.5, 0.055))
    )

    rots = (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (-60, -55, -50, -5)
    )

    max_ts = (15, 30, 60)

    for i, k in enumerate(ks):
        for j, (x_idx, eps) in enumerate(zip(x_idxs[i], epsilons[i])):
            x = xs[x_idx]
            fracs = np.cumsum(results[k, True, eps].T, axis=0) / (trials * n)
            pred = preds[i](x, eps)
            axes[i].axhline(pred, c=colors[i][j], label=labels[i] if j == 0 else '')
            axes[i].plot(np.arange(max_ts[i]+1), fracs[x_idx, :max_ts[i]+1], '.', c=colors[i][j], label='simulation' if j == 0 else '')
            axes[i].text(*text_pos[i][j], f'$x = {x:.2g}, \epsilon = {format_eps(eps)}$', c=colors[i][j], fontsize=8, ha='right', rotation=rots[i][j])

        axes[i].set_yscale('log')
        axes[i].set_xlabel('t')
        axes[i].legend(fontsize=9, bbox_to_anchor=(0.005, 0.12) if k == 4 else (0.36, 0.27), loc='center left')
        axes[i].set_title(f'$k={k}$')

    axes[0].set_ylabel('$F_{{k, t}}(x)$')
    # plt.show()
    plt.savefig('plots/pred-vs-sim-noisy.pdf', bbox_inches='tight')
    plt.close()

def format_eps(eps):
    if eps < 0.01:
        return f'10^{{{np.log10(eps):.0f}}}'
    else:
        return f'{eps:.2g}'

if __name__ == '__main__':
    os.makedirs('plots/', exist_ok=True)

    plot_noisy_convergence()

    # plot_cdf_bounds()


    # plot_small_k_no_noise('small-k-eps-range-50-trials', epsilon=0)

    # plot_small_k_no_noise('small-k-eps-range-50-trials', epsilon=0.01)
    # plot_small_k_no_noise('small-k-eps-range-1-trial', epsilon=0)
    # plot_small_k_no_noise('small-k-eps-range-1-trial', epsilon=0.01)

    # plot_small_k_no_noise('small-k-eps-range-symmetry-50-trials', epsilon=0, symmetry=True)
    # plot_small_k_no_noise('small-k-eps-range-symmetry-50-trials', epsilon=0.01, symmetry=True)
    # plot_small_k_no_noise('small-k-eps-range-symmetry-1-trial', epsilon=0, symmetry=True)
    # plot_small_k_no_noise('small-k-eps-range-symmetry-1-trial', epsilon=0.01, symmetry=True)

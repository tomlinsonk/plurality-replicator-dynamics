import os
import signal
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib
from scipy import stats


# avoid type 3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# allow keyboard interrupt to close pyplot
signal.signal(signal.SIGINT, signal.SIG_DFL)


def plot_pdfs(plot_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 2))

    beta_2 = stats.beta(2, 2)
    beta_half = stats.beta(0.5, 0.5)
    dweibull = stats.dweibull(c=4, loc=0.5, scale=0.3)

    # plot beta_2
    x = np.linspace(beta_2.ppf(0.01), beta_2.ppf(0.99), 100)
    pdf = beta_2.pdf(x)
    axes[0].plot(x, pdf, "b-", lw=3, label="Beta(2,2) pdf")
    axes[0].title.set_text("Beta(2,2) PDF")
    axes[0].fill_between(x, pdf, alpha=0.5)
    axes[0].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # plot beta_half
    x = np.linspace(beta_half.ppf(0.01), beta_half.ppf(0.99), 100)
    pdf = beta_half.pdf(x)
    axes[1].plot(x, pdf, "b-", lw=3, label="Beta(0.5,0.5) pdf")
    axes[1].title.set_text("Beta(0.5,0.5) PDF")
    axes[1].fill_between(x, pdf, alpha=0.5)

    # plot dweibull
    x = np.linspace(dweibull.ppf(0.01), dweibull.ppf(0.99), 100)
    pdf = dweibull.pdf(x)
    axes[2].plot(x, pdf, "b-", lw=3, label="dweibull pdf")
    axes[2].title.set_text("Double Weibull(4,0.5,0.3) PDF")
    axes[2].fill_between(x, pdf, alpha=0.5)
    axes[2].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.savefig(f"plots/{plot_name}.pdf", bbox_inches="tight", dpi=500)
    plt.close()


if __name__ == "__main__":
    plot_pdfs("voter_dsn_pdfs")

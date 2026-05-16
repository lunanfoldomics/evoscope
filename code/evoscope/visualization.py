import matplotlib.pyplot as plt


def plot_genes(sim, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(14, 8))

    for gene, values in sim.gene_history.items():
        ax.plot(values, label=gene, linewidth=2.5)

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Mean expression")
    ax.set_title("Global gene dynamics")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_clusters(sim, gene="T1", save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(14, 8))

    for cid in range(8):
        values = sim.cluster_gene_history[cid][gene]
        if all(v == 0 for v in values):
            continue
        ax.plot(values, label=f"C{cid}", linewidth=2.5)

    ax.set_xlabel("Simulation step")
    ax.set_ylabel(f"{gene} expression")
    ax.set_title(f"{gene} dynamics per cluster")
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_grid_array(grid_array, title="Evoscope grid", show=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid_array, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("q")
    ax.set_ylabel("r")
    fig.colorbar(im, ax=ax, label="state / cluster")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

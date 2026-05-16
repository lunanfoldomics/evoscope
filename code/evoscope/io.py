import csv


def save_global_gene_csv(sim, path="global_genes.csv"):
    genes = list(sim.gene_history.keys())
    if not genes:
        return
    steps = len(sim.gene_history[genes[0]])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + genes)

        for i in range(steps):
            row = [i] + [sim.gene_history[g][i] for g in genes]
            writer.writerow(row)


def save_cluster_gene_csv(sim, path="cluster_genes.csv"):
    genes = ["T1", "T2", "I", "R", "M", "K", "S"]
    steps = len(sim.cluster_gene_history[0][genes[0]])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["step", "cluster", "size"] + genes
        writer.writerow(header)

        for i in range(steps):
            for cid in range(8):
                row = [i, cid, sim.cluster_size_history[cid][i]]
                for g in genes:
                    row.append(sim.cluster_gene_history[cid][g][i])
                writer.writerow(row)

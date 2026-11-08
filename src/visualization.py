"""Publication-style figures. Every function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import TwoSlopeNorm

BLUE, ORANGE, GREEN, GREY = "#1f4e79", "#d9822b", "#2e8b57", "#555555"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "font.size": 12, "axes.titlesize": 14,
        "axes.labelsize": 12, "legend.fontsize": 11, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False, "figure.autolayout": False,
    })


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _subtitle(interval: str, edge_mode: str) -> str:
    return f"{interval} snapshots, {edge_mode} edges"


# ---- 1-4: graph evolution -------------------------------------------------- #
def _line_plot(x, series: dict, title, ylabel, path, interval, edge_mode, colors=None):
    set_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, (label, y) in enumerate(series.items()):
        ax.plot(x, y, marker="o", ms=4, lw=1.8, label=label,
                color=(colors or [BLUE, ORANGE])[i])
    ax.set_xlabel("Days since first email in dataset")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n({_subtitle(interval, edge_mode)})")
    if len(series) > 1:
        ax.legend(frameon=False)
    return _save(fig, path)


def plot_edges(feat, path, interval, edge_mode):
    return _line_plot(feat["start_day"], {"Distinct directed edges": feat["edge_count"]},
                      "Number of directed edges over time", "Distinct directed edges", path,
                      interval, edge_mode)


def plot_density(feat, path, interval, edge_mode):
    return _line_plot(feat["start_day"], {"Density": feat["density"]},
                      "Graph density over time", "Density = edges / (n(n-1))", path,
                      interval, edge_mode)


def plot_active_nodes(feat, path, interval, edge_mode):
    return _line_plot(feat["start_day"], {"Active senders": feat["active_senders"],
                                          "Active receivers": feat["active_receivers"]},
                      "Active senders and receivers over time", "Number of nodes", path,
                      interval, edge_mode)


def plot_mean_degrees(feat, path, interval, edge_mode):
    unit = "distinct neighbours" if edge_mode == "binary" else "emails"
    return _line_plot(feat["start_day"], {"Mean out-degree": feat["mean_out_degree"],
                                          "Mean in-degree": feat["mean_in_degree"]},
                      "Mean in- and out-degree over time", f"Mean degree ({unit} per node)", path,
                      interval, edge_mode)


def plot_clustering(feat, path, interval, edge_mode):
    return _line_plot(feat["start_day"],
                      {"Directed average clustering (Fagiolo)":
                       feat["directed_average_clustering_fagiolo"]},
                      "Directed clustering over time", "Average directed clustering coefficient",
                      path, interval, edge_mode)


# ---- 5-7: embeddings ------------------------------------------------------- #
def plot_embedding(scores, xlabel, ylabel, title, path, color=None, color_label=None,
                   annotate_note=None):
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    if color is None:
        ax.scatter(scores[:, 0], scores[:, 1], s=26, alpha=0.75, color=BLUE, edgecolor="white", lw=0.3)
    else:
        sc = ax.scatter(scores[:, 0], scores[:, 1], s=28, alpha=0.85, c=color, cmap="viridis",
                        edgecolor="white", lw=0.3)
        fig.colorbar(sc, ax=ax, label=color_label)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if annotate_note:
        ax.text(0.02, 0.02, annotate_note, transform=ax.transAxes, fontsize=9, color=GREY)
    return _save(fig, path)


def plot_side_by_side(c_scores, q_scores, c_var, q_frac, color, path, n_samples):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, s, xl, yl, ttl in (
        (axes[0], c_scores, f"PC1 ({c_var[0]*100:.1f}% variance)", f"PC2 ({c_var[1]*100:.1f}% variance)",
         "Classical PCA (same 2 features, same nodes)"),
        (axes[1], q_scores, f"Kernel-PC1 ({q_frac[0]*100:.1f}% of kernel eigenvalue sum)",
         f"Kernel-PC2 ({q_frac[1]*100:.1f}% of kernel eigenvalue sum)",
         "Quantum-kernel PCA (simulated)")):
        sc = ax.scatter(s[:, 0], s[:, 1], c=color, cmap="viridis", s=34, edgecolor="white", lw=0.3)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
    fig.colorbar(sc, ax=axes, label="log(1 + mean total degree)", shrink=0.9)
    fig.suptitle(f"Sample-matched comparison ({n_samples} nodes). Axes/variances are not equivalent.",
                 fontsize=13)
    return _save(fig, path)


def plot_graph_embedding(scores, node_ids, title, path):
    """Plot centered graph-matrix PCA scores for every node in the mapping."""
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=np.arange(len(node_ids)),
                         cmap="viridis", s=22, alpha=0.82, edgecolor="white", lw=0.25)
    fig.colorbar(scatter, ax=ax, label="Stable node index (all mapped nodes)")
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.set_title(title)
    return _save(fig, path)


def plot_graph_reconstruction_sweep(sweep, path):
    """Plot graph-matrix reconstruction error as the retained PC count grows."""
    set_style()
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    colors = {"adjacency": BLUE, "laplacian": ORANGE, "combined": GREEN}
    for name in ("adjacency", "laplacian", "combined"):
        rows = sweep[sweep["representation"] == name].sort_values("n_components")
        ax.plot(rows["n_components"],
                rows["relative_reconstruction_error"] * 100,
                marker="o", lw=2, color=colors[name], label=name.title())
    ax.set_xlabel("Number of retained principal components")
    ax.set_ylabel("Relative reconstruction error (%)")
    ax.set_title("Reconstruction error drops as more components are retained")
    ax.set_xticks(sorted(sweep["n_components"].unique()))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    return _save(fig, path)


# ---- 8-10: subset visualisations ------------------------------------------ #
def top_degree_subset(A, k: int) -> np.ndarray:
    """Indices of the k nodes with largest in+out degree in ``A`` (ties -> lower index)."""
    total = np.asarray(A.sum(axis=1)).ravel() + np.asarray(A.sum(axis=0)).ravel()
    k = min(k, A.shape[0])
    order = np.lexsort((np.arange(len(total)), -total))
    return np.sort(order[:k])


def plot_adjacency_subset(A_sub, ids, title, path, edge_mode):
    set_style()
    D = A_sub.toarray()
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    im = ax.imshow(D, cmap="Blues", vmin=0, vmax=max(D.max(), 1.0))
    ax.set_xticks(range(len(ids))); ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=90, fontsize=7); ax.set_yticklabels(ids, fontsize=7)
    ax.grid(False)
    ax.set_xlabel("Receiver (original node ID)")
    ax.set_ylabel("Sender (original node ID)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Edge weight" if edge_mode == "count" else "Edge present (1)")
    return _save(fig, path)


def plot_directed_graph_subset(A_sub, ids, title, path):
    set_style()
    D = A_sub.toarray()
    G = nx.DiGraph()
    G.add_nodes_from(range(len(ids)))
    for i, j in zip(*np.nonzero(D)):
        G.add_edge(int(i), int(j), weight=float(D[i, j]))
    pos = nx.circular_layout(G)
    deg = np.array([G.in_degree(v) + G.out_degree(v) for v in G.nodes])
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.axis("off")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300 + 60 * deg, node_color=deg, cmap="YlOrBr", vmin=0,
                           vmax=max(deg.max(), 1) * 2.0, edgecolors=GREY)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=12,
                           edge_color=BLUE, alpha=0.55, width=1.0,
                           connectionstyle="arc3,rad=0.12", node_size=300 + 60 * deg)
    nx.draw_networkx_labels(G, pos, labels={i: str(v) for i, v in enumerate(ids)}, font_size=8, ax=ax)
    ax.set_title(title)
    return _save(fig, path)


def plot_laplacian_subset(L_sub, ids, title, path):
    set_style()
    D = L_sub.toarray()
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    if D.min() < 0 < D.max():
        norm = TwoSlopeNorm(vmin=D.min(), vcenter=0.0, vmax=D.max())
        im = ax.imshow(D, cmap="RdBu_r", norm=norm)
    else:
        im = ax.imshow(D, cmap="RdBu_r")
    ax.set_xticks(range(len(ids))); ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=90, fontsize=8); ax.set_yticklabels(ids, fontsize=8)
    ax.grid(False)
    if len(ids) <= 16:
        for i in range(len(ids)):
            for j in range(len(ids)):
                if D[i, j] != 0:
                    ax.text(j, i, f"{D[i, j]:g}", ha="center", va="center", fontsize=6.5)
    ax.set_xlabel("Column node (original ID)")
    ax.set_ylabel("Row node (original ID)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="L_out entry (diag = out-degree, off-diag = -A)")
    return _save(fig, path)

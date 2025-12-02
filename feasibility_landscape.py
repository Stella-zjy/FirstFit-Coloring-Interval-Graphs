from __future__ import annotations

import os
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import PowerNorm
from tqdm import tqdm

from generator import to_frac, u_sequence, find_termination_index


def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def terminates_delta0(r: Fraction, theta: Fraction, n_max: int) -> Tuple[bool, Optional[int]]:
    """Return (terminates?, peak index N if any) for delta=0."""
    u = u_sequence(r=r, theta=theta, delta=Fraction(0), n_max=n_max)
    N = find_termination_index(u)
    return (N is not None), N


def compute_feasibility_grid(
    r_vals: List[float],
    theta_vals: List[float],
    n_max: int,
    desc: str,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Returns:
      feasible[i][j] in {0,1} for theta_vals[i], r_vals[j]
      peakN[i][j] = N if feasible else -1
    """
    feasible: List[List[int]] = []
    peakN: List[List[int]] = []

    for th in tqdm(theta_vals, desc=desc, unit="row"):
        thF = to_frac(th)
        row_f: List[int] = []
        row_n: List[int] = []

        for rr in r_vals:
            rF = to_frac(rr)
            ok, N = terminates_delta0(r=rF, theta=thF, n_max=n_max)
            row_f.append(1 if ok else 0)
            row_n.append(int(N) if ok and N is not None else -1)

        feasible.append(row_f)
        peakN.append(row_n)

    return feasible, peakN


def plot_feasibility_heatmap(
    feasible: List[List[int]],
    r_vals: List[float],
    theta_vals: List[float],
    title: str,
    savepath: str,
    mark_point: Tuple[float, float] = (5.0, 2.0),
):
    """
    Square figure, binary colors, no colorbar:
      0 = divergent (grey)
      1 = terminated (blue)
    """
    cmap = ListedColormap(["#BDBDBD", "#1F77B4"])  # grey, blue
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(12, 12))  # square canvas

    ax.imshow(
        feasible,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        extent=[min(r_vals), max(r_vals), min(theta_vals), max(theta_vals)],
        cmap=cmap,
        norm=norm,
    )

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(title)

    # Legend: blue for terminated, grey for divergent
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#1F77B4", edgecolor="black", label="terminated"),
        Patch(facecolor="#BDBDBD", edgecolor="black", label="divergent"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.8,
        fancybox=True,
        fontsize=28,          # larger text
        handlelength=2,     # larger color patches
        handleheight=1.2,
        borderpad=1.0,
        labelspacing=0.8,
    )

    # Mark (5,2) if within view
    rx, tx = mark_point
    if (min(r_vals) <= rx <= max(r_vals)) and (min(theta_vals) <= tx <= max(theta_vals)):
        ax.scatter([rx], [tx], marker="x", s=300, linewidths=3.0, color="white")

    # Keep identical output pixel size across figures by NOT using bbox_inches="tight"
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


def plot_peakN_heatmap(
    peakN: List[List[int]],
    r_vals: List[float],
    theta_vals: List[float],
    title: str,
    savepath: str,
    mark_point: Tuple[float, float] = (5.0, 2.0),
):
    """
    Square figure, colorbar, peak index N where termination happens.
    Divergent cells (-1) are shown in grey.
    """
    arr = np.array(peakN, dtype=float)
    arr[arr < 0] = np.nan  # divergent -> NaN so we can color as 'bad'

    cmap = plt.get_cmap("Purples").copy()
    cmap.set_bad(color="#BDBDBD")  # divergent cells as grey

    fig, ax = plt.subplots(figsize=(12, 12))  # square canvas

    gamma = 0.3
    im = ax.imshow(
        arr,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        extent=[min(r_vals), max(r_vals), min(theta_vals), max(theta_vals)],
        cmap=cmap,
        norm=PowerNorm(gamma=gamma),
    )

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(title)

    # Mark (5,2) if within view
    rx, tx = mark_point
    if (min(r_vals) <= rx <= max(r_vals)) and (min(theta_vals) <= tx <= max(theta_vals)):
        ax.scatter([rx], [tx], marker="x", s=300, linewidths=3.0, color="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Peak index $N$", fontsize=26)
    cbar.ax.tick_params(labelsize=24)

    # Keep identical output pixel size across figures by NOT using bbox_inches="tight"
    fig.savefig(savepath, dpi=220)
    plt.close(fig)


def main():
    os.makedirs("plots", exist_ok=True)

    # Make all words/labels/titles much larger (applies to all four figures)
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 28,
            "axes.labelsize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
        }
    )

    # -----------------------
    # Coarse landscape
    # -----------------------
    r_min, r_max = 4.2, 5.2
    th_min, th_max = 1.2, 2.2

    r_points = 101
    th_points = 101
    n_max = 50

    r_vals = linspace(r_min, r_max, r_points)
    theta_vals = linspace(th_min, th_max, th_points)

    feasible, peakN = compute_feasibility_grid(
        r_vals, theta_vals, n_max=n_max, desc=f"Coarse grid (n_max={n_max})"
    )

    plot_feasibility_heatmap(
        feasible,
        r_vals=r_vals,
        theta_vals=theta_vals,
        title=rf"Feasibility landscape",
        savepath="plots/feasibility_landscape_coarse.pdf",
        mark_point=(5.0, 2.0),
    )

    plot_peakN_heatmap(
        peakN,
        r_vals=r_vals,
        theta_vals=theta_vals,
        title=rf"Peak index N (termination)",
        savepath="plots/peakN_landscape_coarse.pdf",
        mark_point=(5.0, 2.0),
    )

    # -----------------------
    # Zoom near r=5
    # -----------------------
    r_min2, r_max2 = 4.90, 5.02
    th_min2, th_max2 = 1.90, 2.10

    r_points2 = 101
    th_points2 = 101
    n_max2 = 100

    r_vals2 = linspace(r_min2, r_max2, r_points2)
    theta_vals2 = linspace(th_min2, th_max2, th_points2)

    feasible2, peakN2 = compute_feasibility_grid(
        r_vals2, theta_vals2, n_max=n_max2, desc=f"Zoom grid (n_max={n_max2})"
    )

    plot_feasibility_heatmap(
        feasible2,
        r_vals=r_vals2,
        theta_vals=theta_vals2,
        title=rf"Feasibility landscape near $r=5$",
        savepath="plots/feasibility_landscape_zoom.pdf",
        mark_point=(5.0, 2.0),
    )

    plot_peakN_heatmap(
        peakN2,
        r_vals=r_vals2,
        theta_vals=theta_vals2,
        title=rf"Peak index N (termination) near $r=5$",
        savepath="plots/peakN_landscape_zoom.pdf",
        mark_point=(5.0, 2.0),
    )

    print("Saved plots:")
    print("  - plots/feasibility_landscape_coarse.png")
    print("  - plots/peakN_landscape_coarse.png")
    print("  - plots/feasibility_landscape_zoom.png")
    print("  - plots/peakN_landscape_zoom.png")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple
import math

import matplotlib.pyplot as plt


# ---------------------------
# Helpers
# ---------------------------

def to_frac(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, float):
        return Fraction(x).limit_denominator(100000)
    if isinstance(x, str):
        x = x.strip()
        if "/" in x:
            a, b = x.split("/")
            return Fraction(int(a.strip()), int(b.strip()))
        return Fraction(x)
    raise TypeError(f"Cannot convert {type(x)} to Fraction")


def fmt_decimal(val: Fraction, precision: int = 5) -> str:
    """Helper to format a Fraction as a string with decimal approximation."""
    return f"{val} (~{float(val):.{precision}f})"

# ---------------------------
# Recurrence
# ---------------------------

def u_sequence(r: Fraction, theta: Fraction, delta: Fraction, n_max: int) -> List[Fraction]:
    """
    u0 = u1 = 1
    u2 = theta + delta
    u_n = (r-theta)u_{n-1} - (r-2theta)u_{n-2} - theta u_{n-3}, for n>=3
    """
    r = to_frac(r); theta = to_frac(theta); delta = to_frac(delta)
    if n_max < 2:
        raise ValueError("n_max must be >= 2")

    u = [Fraction(1), Fraction(1), theta + delta]
    for n in range(3, n_max + 1):
        un = (r - theta) * u[n - 1] - (r - 2 * theta) * u[n - 2] - theta * u[n - 3]
        u.append(un)
    return u


def find_termination_index(u: List[Fraction]) -> Optional[int]:
    """
    Return the smallest N such that
        u0 = u1 <= u2 < u3 < ... < uN
    and
        uN >= u_{N+1}
    with all u2..u_{N+1} > 0.
    """
    if len(u) < 4:
        return None
    if u[0] != u[1]:
        return None
    if u[1] > u[2] or u[2] <= 0:
        return None

    for n in range(2, len(u) - 1):
        if u[n] <= 0 or u[n + 1] <= 0:
            return None
        if u[n + 1] > u[n]:
            continue
        return n  # first non-increase = peak index

    return None


def terminating_prefix(r: Fraction, theta: Fraction, delta: Fraction, n_max: int) -> Optional[Tuple[int, List[Fraction]]]:
    """
    Return (N, [u0..u_{N+1}]) if termination condition holds, else None.
    """
    u = u_sequence(r, theta, delta, n_max=n_max)
    N = find_termination_index(u)
    if N is None:
        return None
    return N, u[:N + 2]


# ---------------------------
# Delta selection (grid search)
# ---------------------------

def candidate_deltas(max_delta: Fraction, denom_max: int) -> List[Fraction]:
    """
    Enumerate rational deltas p/q <= max_delta (dense-ish), descending order.
    """
    max_delta = to_frac(max_delta)
    if max_delta <= 0:
        return []
    cands = set()
    for q in range(1, denom_max + 1):
        p_max = int(math.floor(float(max_delta * q)))
        for p in range(1, p_max + 1):
            cands.add(Fraction(p, q))
    return sorted(cands, reverse=True)


def find_good_delta_grid(
    r: Fraction,
    theta: Fraction,
    max_delta: Fraction,
    denom_max: int,
    n_max: int
) -> Tuple[Fraction, int, List[Fraction]]:
    """
    Find a delta by rational-grid search that yields a terminating u-sequence.
    Returns (delta, N, u_prefix[0..N+1]).
    """
    for d in candidate_deltas(max_delta=max_delta, denom_max=denom_max):
        res = terminating_prefix(r, theta, d, n_max=n_max)
        if res is not None:
            N, up = res
            return d, N, up
    raise ValueError("No feasible delta found in the specified rational grid.")


# ---------------------------
# Iteration driver
# ---------------------------

@dataclass(frozen=True)
class StepLog:
    step: int
    r: Fraction
    theta_before: Fraction
    delta: Fraction
    theta_after: Fraction
    N: int
    u_prefix: List[Fraction]


def iterate_gap_reduction(
    r: Fraction,
    theta0: Fraction = Fraction(1),
    eps: Fraction = Fraction(1, 100),
    max_steps: int = 50,
    denom_max: int = 50,
    n_max: int = 1000,
    verbose: bool = True,
) -> List[StepLog]:
    """
    Iteratively increase theta toward r-2 by repeatedly choosing delta such that u-sequence terminates.
    Stop when gap = (r-2)-theta <= eps, or after max_steps.
    """
    r = to_frac(r); theta = to_frac(theta0); eps = to_frac(eps)
    logs: List[StepLog] = []

    for t in range(1, max_steps + 1):
        gap = (r - 2) - theta
        if gap <= eps:
            if verbose:
                print(f"[stop] theta={theta} gap={gap} <= eps={eps}")
            break

        max_delta = gap  # don't overshoot r-2
        d, N, up = find_good_delta_grid(r, theta, max_delta, denom_max=denom_max, n_max=n_max)

        theta_after = theta + d
        logs.append(StepLog(
            step=t, r=r, theta_before=theta, delta=d, theta_after=theta_after, N=N, u_prefix=up
        ))

        if verbose:
            print(f"[step {t}] theta={theta} gap={gap} -> delta={d} => theta'={theta_after} (N={N})")

        theta = theta_after

    return logs


# ---------------------------
# Visualization
# ---------------------------

def plot_u(u: List[Fraction], N: Optional[int] = None, title: Optional[str] = None,
           show: bool = True, savepath: Optional[str] = None):
    """
    Plot u_n values. If N is given, mark the peak u_N and the turnover point u_{N+1}.
    """
    xs = list(range(len(u)))
    ys = [float(v) for v in u]

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot line and points
    ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=12)
    
    # Increase label sizes
    ax.set_xlabel("n", fontsize=24)
    ax.set_ylabel(r"$u_n$", fontsize=24)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    ax.grid(True, linewidth=0.3, alpha=0.5)

    if N is not None and 0 <= N < len(u) - 1:
        ax.scatter([N, N + 1], [ys[N], ys[N + 1]], s=120, zorder=5) # Larger scatter points
        ax.axvline(N, linestyle="--", linewidth=2)
        
        # Increase annotation text size
        ax.text(N, ys[N], f"  peak N={N}", va="bottom", fontsize=24, fontweight='bold')
        ax.text(N + 1, ys[N + 1], "  N+1", va="bottom", fontsize=24)

    if title:
        # Increase title size
        ax.set_title(title, fontsize=24, pad=15)

    # Tight layout helps ensure large labels don't get cut off
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------
# Demo run
# ---------------------------

if __name__ == "__main__":
    # Example: r=4.5
    r = Fraction(9, 2)
    theta0 = Fraction(1)

    # If eps=0, it won't stop early; it will just do max_steps.
    eps = Fraction(0, 1)

    logs = iterate_gap_reduction(
        r=r,
        theta0=theta0,
        eps=eps,
        max_steps=50,
        denom_max=50,
        n_max=100,
        verbose=True
    )

    # Plot the u-prefix at each step
    p = 4
    for log in logs:
        plot_u(
            log.u_prefix,
            N=log.N,
            title=(
                f"Step {log.step}: "
                f"r={float(log.r)}, "
                f"θ={float(log.theta_before)} → {float(log.theta_after)}, "
                f"δ={float(log.delta)}, "
                f"N={log.N}"
            ),
            show = False,
            savepath=f"plots/u_seq_r_{float(log.r)}_step_{log.step}.png"
        )

    if logs:
        last = logs[-1]
        print("\n" + "="*40)
        print("FINAL RESULTS")
        print("="*40)
        print(f"Target Ratio (r): {float(last.r)}")
        print(f"Final Theta     : {fmt_decimal(last.theta_after)}")
        print(f"Last Delta Used : {fmt_decimal(last.delta)}")
        print(f"Sequence Length : {last.N + 2} terms")
        print("-" * 40)
        
        # Print sequence in both formats for verification against paper examples
        print("Sequence u_n (Decimal):")
        print([f"{float(x):.4f}" for x in last.u_prefix])
        
        print("\nSequence u_n (Fraction):")
        print([str(x) for x in last.u_prefix])
        print("="*40)

        # Plot the final sequence
        plot_u(
            last.u_prefix,
            N=last.N,
            title=f"Final Sequence: r={float(last.r)}, Theta={float(last.theta_after):.2f} (Peak at N={last.N})",
            show = False,
            savepath=f"plots/u_seq_final_r_{float(last.r)}.png"
        )
    else:
        print("No successful steps performed.")
from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


def gradient_descent_map(x, eta):
    """Gradient descent map for f(x) = (x^2 - 1)^2."""
    x = np.asarray(x, dtype=float)
    return (1 + 4 * eta) * x - 4 * eta * x**3


def derivative_of_map(x, eta):
    """Derivative of the gradient descent map."""
    x = np.asarray(x, dtype=float)
    return 1 + 4 * eta - 12 * eta * x**2


@dataclass
class LyapunovExperimentConfig:
    """Configuration for scanning the Lyapunov exponent over eta values."""

    etas: np.ndarray | None = None
    x0: float = 0.123456789
    n_burn: int = 2000
    n_iter: int = 5000
    escape_radius: float = 1e6
    eps: float = 1e-300

    def __post_init__(self) -> None:
        if self.etas is None:
            self.etas = np.linspace(0.001, 0.8, 3000)
        else:
            self.etas = np.asarray(self.etas, dtype=float)


def lyapunov_exponent(
    eta,
    x0=0.123456789,
    n_burn=2000,
    n_iter=5000,
    escape_radius=1e6,
    eps=1e-300,
):
    """Estimate the Lyapunov exponent of the map from one initial point."""
    x = x0

    for _ in range(n_burn):
        x = gradient_descent_map(x, eta)
        if not np.isfinite(x) or abs(x) > escape_radius:
            return np.nan

    s = 0.0
    for _ in range(n_iter):
        deriv = abs(derivative_of_map(x, eta))
        s += np.log(max(deriv, eps))

        x = gradient_descent_map(x, eta)
        if not np.isfinite(x) or abs(x) > escape_radius:
            return np.nan

    return s / n_iter


def run_lyapunov_scan(config: LyapunovExperimentConfig):
    """Run a Lyapunov exponent scan for all eta values in the configuration."""
    lambdas = np.array(
        [
            lyapunov_exponent(
                eta,
                x0=config.x0,
                n_burn=config.n_burn,
                n_iter=config.n_iter,
                escape_radius=config.escape_radius,
                eps=config.eps,
            )
            for eta in config.etas
        ]
    )
    return config.etas, lambdas


def plot_lyapunov_exponent(etas, lambdas, *, show: bool = True):
    """Plot the Lyapunov exponent with a few reference markers."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(etas, lambdas, lw=0.6)

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(1 / 4, linestyle="--", lw=0.8, label=r"$\eta=1/4$")
    ax.axvline(
        (np.sqrt(5) - 1) / 4,
        linestyle="--",
        lw=0.8,
        label=r"$\eta=(\sqrt{5}-1)/4$",
    )
    ax.axvline(1 / 2, linestyle="--", lw=0.8, label=r"$\eta=1/2$")

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"Lyapunov exponent $\lambda$")
    ax.set_title(r"Lyapunov exponent of $F_\eta(x)=(1+4\eta)x-4\eta x^3$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Lyapunov exponent scan for the GD map"
    )
    parser.add_argument("--eta-min", type=float, default=0.001)
    parser.add_argument("--eta-max", type=float, default=0.8)
    parser.add_argument("--eta-count", type=int, default=3000)
    parser.add_argument("--x0", type=float, default=0.123456789)
    parser.add_argument("--n-burn", type=int, default=2000)
    parser.add_argument("--n-iter", type=int, default=5000)
    parser.add_argument("--escape-radius", type=float, default=1e6)
    parser.add_argument("--eps", type=float, default=1e-300)
    return parser

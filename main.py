from __future__ import annotations

import numpy as np

from gd_and_chaos import (
    LyapunovExperimentConfig,
    build_arg_parser,
    plot_lyapunov_exponent,
    run_lyapunov_scan,
)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    config = LyapunovExperimentConfig(
        etas=np.linspace(args.eta_min, args.eta_max, args.eta_count),
        x0=args.x0,
        n_burn=args.n_burn,
        n_iter=args.n_iter,
        escape_radius=args.escape_radius,
        eps=args.eps,
    )
    etas, lambdas = run_lyapunov_scan(config)
    plot_lyapunov_exponent(etas, lambdas, show=True)


if __name__ == "__main__":
    main()

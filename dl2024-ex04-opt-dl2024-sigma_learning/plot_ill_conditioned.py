import argparse

from lib.plot import plot_ill_conditioning

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ill-conditioning.")
    parser.add_argument(
        "-c",
        "--condition_number",
        type=int,
        default=1,
        help="Condition number of the quadratic function matrix. Default: 1.",
    )
    parser.add_argument(
        "-m",
        "--momentum",
        type=float,
        default=0.0,
        help="Momentum of the SGD. Default: 0.0.",
    )
    parser.add_argument(
        "-p",
        "--preconditioning",
        action="store_true",
        default=False,
        help="Apply preconditioning matrix "
        "(inverse hessian of the quadratic function matrix) to the gradient. "
        "Default: False.",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=500,
        help="Number of optimization steps. Default: 500.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.22,
        help="Learning rate of the SGD. Default: 0.22.",
    )
    parser.add_argument(
        "--init",
        nargs="*",
        type=float,
        default=(0.7, 0.7),
        help="Initial point. Default (0.7, 0.7).",
    )
    args = parser.parse_args()

    plot_ill_conditioning(
        condition_number=args.condition_number,
        momentum=args.momentum,
        preconditioning=args.preconditioning,
        steps=args.steps,
        lr=args.lr,
        init=tuple(args.init),
    )

"""
MLE for Birth-Death Trees

For complete sampling at present time, Stadler's likelihood formula is:
L(λ, μ | tree) = (n-1)! × λ^(n-1) × exp(-(λ+μ)×S) / (λ - μ×exp(-(λ-μ)×T))^n

Where:
- n = number of tips
- S = sum of all branch lengths (excluding root)
- T = tree height (time from root to present)

"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ete3 import Tree
import os
import glob
from mle.ldbp_exact import extract_lbdp_stats_from_tree, lbdp_exact_python
from pathlib import Path
import tqdm

# Import for better factorial calculation


def load_tree_from_newick(newick_file):
    """Load a phylogenetic tree from a Newick file."""
    try:
        with open(newick_file, "r") as f:
            newick_str = f.read().strip()
        tree = Tree(newick_str, format=1)
        return tree
    except Exception as e:
        print(f"Error loading tree from {newick_file}: {e}")
        return None


def birth_death_likelihood(tree, **kwargs):
    """
    Calculate negative log-likelihood for birth-death model.

    Parameters:
    - params: [lambda, mu] (birth rate, death rate)
    - tree: ete3 Tree object
    - penalty_weight: Weight for penalty term to prevent unrealistic rates (default 1.0)

    Returns: negative log-likelihood (for minimization)
    """
    stats = extract_lbdp_stats_from_tree(tree)
    ll = lbdp_exact_python(**kwargs, stats=stats)
    return ll


def estimate_mle_birth_death(tree):
    """
    Estimate birth and death rates using Maximum Likelihood Estimation.

    Uses Stadler's exact likelihood formula and robust optimization to find
    the true MLE estimates.

    Returns: (lambda_mle, mu_mle, R0_mle) where R0 = lambda/mu
    """
    if tree is None:
        return None, None, None

    n_tips = len(tree.get_leaves())
    if n_tips < 2:
        return None, None, None

    # Get tree statistics
    tree_stats = extract_lbdp_stats_from_tree(tree)
    if tree_stats is None:
        return None, None, None

    lambda_init, mu_init = np.random.uniform(0.1, 1, 2)

    # Ensure μ < λ (typically true for growing populations)
    if mu_init >= lambda_init:
        mu_init = lambda_init * 0.8

    # Try optimization with L-BFGS-B (gradient-based, handles bounds)
    # best_result = None
    # best_ll = np.inf

    def objective_function(params):
        lambda_, mu_ = params
        if mu_ >= lambda_:
            return np.inf
        return -birth_death_likelihood(
            tree, lambda_=lambda_, mu=mu_, psi=0.5, n0=1
        )  # lock psi, n0, and the tree

    # This prevents numerical differentiation from trying invalid values
    result = minimize(
        objective_function,
        [lambda_init, mu_init],
        method="L-BFGS-B",
        bounds=[(1e-3, 1), (1e-3, 1)],
        options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-5, "maxls": 50},
    )
    return result.x[0], result.x[1], result.fun


def main():
    """Main function to estimate MLE parameters for all trees."""
    print("=" * 60)
    print("MLE Estimation for Birth-Death Trees")
    print("=" * 60)

    # Paths
    base_dir = Path(__file__).resolve().parent.parent
    trees_dir = base_dir / "output_trees"
    params_file = base_dir / "all_params.csv"
    output_dir = base_dir / "mle_results_bldp_exact"

    # Create output directory
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "mle_estimates.csv"

    # Load true parameters
    print("\nLoading true parameters...")
    true_params = pd.read_csv(params_file)
    print(f"Loaded {len(true_params)} parameter sets")

    # Get all tree files
    tree_files = sorted(glob.glob(os.path.join(trees_dir, "tree_*.nwk")))
    print(f"Found {len(tree_files)} tree files")

    # Create results storage
    results = []

    # Process trees
    print("\nProcessing trees with MLE...")
    for tree_file in tqdm.tqdm(tree_files, unit="tree"):
        # Extract tree index from filename
        tree_idx = int(
            os.path.basename(tree_file).replace("tree_", "").replace(".nwk", "")
        )

        # Get true parameters for this tree
        true_row = true_params[true_params["idx"] == tree_idx]
        if len(true_row) == 0:
            continue

        true_mu = true_row["psi_I"].values[0]
        true_lambda = true_row["la_II"].values[0]
        n_tips = int(true_row["tips"].values[0])

        # Load tree
        tree = load_tree_from_newick(tree_file)
        if tree is None:
            continue

        # Estimate MLE
        lambda_mle, mu_mle, likelihood = estimate_mle_birth_death(tree)

        # Store results
        result = {
            "tree_idx": tree_idx,
            "n_tips": n_tips,
            "true_mu": true_mu,
            "true_lambda": true_lambda,
            "mle_mu": mu_mle if mu_mle is not None else np.nan,
            "mle_lambda": lambda_mle if lambda_mle is not None else np.nan,
            "ll": likelihood if likelihood is not None else np.nan,
        }
        results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print(f"\nSuccessfully processed {len(results_df)} trees")

    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"Saved MLE estimates to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("MLE ESTIMATION SUMMARY")
    print("=" * 60)
    print(f"Total trees analyzed: {len(results_df)}")
    print(f"Successful estimates: {results_df['mle_mu'].notna().sum()}")

    # Print statistics
    if results_df["mle_mu"].notna().sum() > 0:
        print(f"\nMLE mu statistics:")
        print(f"  Mean: {results_df['mle_mu'].mean():.4f}")
        print(f"  Std: {results_df['mle_mu'].std():.4f}")
        print(
            f"  Range: [{results_df['mle_mu'].min():.4f}, {results_df['mle_mu'].max():.4f}]"
        )
        print(f"\nMLE lambda (μ) statistics:")
        print(f"  Mean: {results_df['mle_lambda'].mean():.4f}")
        print(f"  Std: {results_df['mle_lambda'].std():.4f}")
        print(
            f"  Range: [{results_df['mle_lambda'].min():.4f}, {results_df['mle_lambda'].max():.4f}]"
        )

    print(f"\nResults saved to: {output_file}")
    print("\nMLE estimation complete!")


if __name__ == "__main__":
    # test_tree = load_tree_from_newick("output_trees/tree_5.nwk")
    # result = estimate_mle_birth_death(test_tree)
    main()

# ChatGPT generated and modified
from ete3 import Tree
from math import lgamma
import numpy as np
from collections import namedtuple

Event = namedtuple("Event", ["time", "code", "lineages"])


"""
Notes:

No interval nodes
We're 99% sure that this process from tree simmulator reflects heterochronous sampling (for this particular birth death model)
psi = sampling rate
mu is generally per capita deaths
lambda is per capita births

The birth-death process in phylopomp is not the same as the one in the Phylodeep

To do this small exercise we needed to put a new model into Phylopomp

regular non linear systems used to be difficult to do likelihood simulation
Is it worth the effort of finding the likelihood;
    - generally the answer has been yes
    - likelihood based inference is efficient from a theoretical perspective, so the question is what is lost through Phylodeep
"""


def compute_node_ages(tree: Tree):
    """
    Given an ete3 Tree with branch lengths = times,
    return:
      - tree_height: time from root to most distant tip
      - ages: dict(node -> age measured from present backwards)
    """
    # distance from root to each node
    heights = {}

    def fill_heights(node, current_length):
        heights[node] = current_length
        for child in node.children:
            fill_heights(child, current_length + child.dist)

    root = tree.get_tree_root()
    fill_heights(root, 0.0)

    # present = most recent tip time
    tree_height = max(heights[leaf] for leaf in tree.iter_leaves())

    # age from present (like tf - time in phylopomp)
    ages = {node: tree_height - h for node, h in heights.items()}
    return tree_height, ages, root


def extract_lbdp_stats_from_tree(tree: Tree):
    """
    Parse a Newick tree and extract the ingredients
    needed for an lbdp_exact-like likelihood.

    Returns a dict with:
      - x0: root age (backwards from present)
      - x: 1D np.array of branching times (coalescent ages)
      - y: 1D np.array of sampling times (tip ages)
      - n: number of roots (usually 1)
      - m: number of 'live' samples   (we'll set = #tips for now)
      - k: number of 'dead' samples   (0 in this simplified mapping)
      - tf: time of most recent sample (tree_height)
    """
    tf, ages, root = compute_node_ages(tree)

    # root time (age from present)
    x0 = ages[root]

    # branching events: internal nodes except the root
    branching_times = []
    for node in tree.traverse():
        if node.is_root():
            continue
        if not node.is_leaf():
            branching_times.append(ages[node])
    x = np.array(sorted(branching_times))

    # sampling events: tips
    sample_times = [ages[leaf] for leaf in tree.iter_leaves()]
    y = np.array(sorted(sample_times))

    # assuming a single rooted tree
    n = 1

    # Now the tricky bit: m vs k.
    # For sampling-at-removal with no other sampling, a natural mapping is:
    #   - all sampled individuals are of one kind.
    # If you want to match lbdp_exact exactly, you have to decide:
    #   either:
    #      m = number of samples (treat as 'live'), k = 0
    #   or:
    #      k = number of samples (treat as 'dead'), m = 0
    #
    # Pick one consistently and adjust the formula accordingly.
    m = len(y)
    k = 0

    return dict(x0=x0, x=x, y=y, n=n, m=m, k=k, tf=tf)


def lbdp_exact_python(lambda_, mu, psi, n0, stats):
    """
    LDBP exact likelihood calculation in Python. This is shamelessly stolen from the
    Phylopomp package, with the initial stats extraction from the tree (the first part of the
    Phylopomop function) moved to a separate function and provided by ChatGPT.
    """
    x0 = stats["x0"]
    x = stats["x"]
    y = stats["y"]
    n = stats["n"]
    k = stats["k"]
    m = stats["m"]

    # f, p0, Q, etc. directly from the R code
    def f(z):
        # handle infinities like the R code (simplified)
        z = np.asarray(z)
        out = (1 - z) / (1 + z)
        # you can add explicit handling of z = ±inf, z = -1 if needed
        return out

    d = np.sqrt((lambda_ - mu - psi) ** 2 + 4 * lambda_ * psi)
    a = (lambda_ + mu + psi) / (2 * lambda_)
    b = d / (2 * lambda_)
    z0 = f((1 - a) / b)

    def p0(t):
        return a + b * f(z0 * np.exp(d * t))

    def Q(t):
        e = np.exp(d * t)
        g = 1 + z0 * e
        return e / (g * g)

    # log choose n0 n = log(n0 choose n) and log(n!)

    log_choose = lgamma(n0 + 1) - lgamma(n + 1) - lgamma(n0 - n + 1)
    log_fact_n = lgamma(n + 1)

    ll = (
        log_choose
        + log_fact_n
        + (n0 - n) * np.log(p0(x0))
        + (m - n) * np.log(2 * lambda_)
        + (k + m) * np.log(psi)
        + n * np.log(Q(x0))
        + np.sum(np.log(Q(x)))
        + np.sum(np.log(p0(y) / Q(y)))
    )
    return ll


tree = Tree(open("output_trees/tree_5.nwk").read().strip())
stats = extract_lbdp_stats_from_tree(tree)
ll = lbdp_exact_python(lambda_=0.7, mu=0.2, psi=0.5, n0=1, stats=stats)

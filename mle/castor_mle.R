library(castor)
library(dplyr)

# read first tree from file produced by TreeSimulator
nwk <- readLines("./output_trees/tree_5.nwk", n = 1)
tree <- read_tree(
    string = nwk,
    include_edge_lengths = TRUE,
    look_for_edge_labels = TRUE
)


# Cool; now do this log likelihood function
bd_loglik_castor <- function(
    tree,
    lambda,
    mu,
    rho = 1.0,
    condition = "crown"
) {
    ll <- loglikelihood_hbd(
        tree = tree,
        oldest_age = NULL, # uses root age
        rho0 = rho,
        age_grid = NULL, # constant rates
        lambda = lambda,
        mu = mu,
        condition = condition
    )

    if (!ll$success) {
        stop(ll$error)
    }
    ll$loglikelihood
}


optimizer <- function(nwk_tree) {
    sub_optim <- function(theta) {
        lambda <- theta[1]
        mu <- theta[2]
        -bd_loglik_castor(
            nwk_tree,
            lambda = lambda,
            mu = mu,
            rho = 0.5,
            condition = "crown"
        )
    }
    sub_optim
}

bd_mle <- function(nwk_str, rho = 0.5, condition = "auto", ...) {
    tree <- read_tree(
        string = nwk_str,
        include_edge_lengths = TRUE,
        look_for_edge_labels = TRUE
    )
    fn <- optimizer(tree)
    optim(
        c(1.0, 1.0), # same starting point every time, because why not
        fn,
        lower = c(0, 0),
        ...
    )
}

all_trees <- list.files(
    "./output_trees",
    pattern = "tree_.*\\.nwk$",
    full.names = TRUE
)

tree_idx <- sapply(
    all_trees,
    function(f) {
        as.numeric(sub(".*tree_(\\d+)\\.nwk$", "\\1", f))
    },
    USE.NAMES = FALSE
)
names(all_trees) <- tree_idx

results <- lapply(all_trees[1:100], function(f) {
    nwk <- readLines(f, n = 1)
    result <- bd_mle(nwk, rho = 0.5, condition = "crown", method = "L-BFGS-B")
})

all_params <- read.csv("./all_params.csv")

data.frame(
    t(sapply(results, function(res) res$par)),
    names = c("la_II_true", "psi_I_true")
)

merge(
    all_params[, c("idx", "la_II", "psi_I")],
)


all_params[order(all_params$idx), c("la_II", "psi_I")][1:10, ]

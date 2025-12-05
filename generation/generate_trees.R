library(parallel)
library(phylopomp)
library(pbapply)

n_trees <- 2000


simulate_one <- function(tf = 10, max_iter = 5) {
    # model name / args might be slightly different, but conceptually:
    ll <- Inf

    i <- 0
    psi = 0.5
    while (is.infinite(ll) & i <= max_iter) {
        i <- i + 1
        lambda = runif(1, 0.1, 1.0)
        mu = runif(1, 0.0, 1.0)

        if (mu > lambda) {
            mu <- lambda * runif(1, 0.0, 0.9)
        }
        g <- simulate(
            "LBDP",
            lambda = lambda,
            mu = mu,
            n0 = 1,
            psi = psi,
            t0 = 0,
            time = tf
        )
        ll <- lbdp_exact(g, lambda = lambda, mu = mu, psi = psi, n0 = 1)
    }

    list(
        tree = g,
        newick = newick(g),
        params = c(lambda = lambda, mu = mu, psi = psi, ntips = nsamples(g))
    )
}


cl <- makeForkCluster(detectCores() - 1)
clusterSetRNGStream(cl, iseed = 20250607)
sim_list <- pbreplicate(n_trees, simulate_one(), simplify = FALSE, cl = cl)
stopCluster(cl)


# write Newicks for PhyloDeep
dir.create("phylopomp_newick", showWarnings = FALSE)
for (i in seq_along(sim_list)) {
    fn <- file.path("phylopomp_newick", paste0("tree_", i, ".nwk"))
    writeLines(sim_list[[i]]$newick, fn)
}

params <- as.data.frame(do.call("rbind", lapply(sim_list, '[[', 'params')))
params$idx <- seq_len(nrow(params))
write.csv(params, "phylopomp_sim_params.csv", row.names = FALSE)


fit_one_mle <- function(g, psi_fixed = NULL, n0 = 1L) {
    fn <- function(par) {
        lambda <- par[1]
        mu <- par[2]

        ll_val <- lbdp_exact(
            g,
            lambda = lambda,
            mu = mu,
            psi = psi_fixed,
            n0 = n0
        )
        if (is.infinite(ll_val) || is.nan(ll_val)) {
            return(1e-8)
        } else {
            return(-ll_val)
        }
    }

    opt <- optim(
        par = c(0.5, 0.1),
        fn = fn,
        method = "L-BFGS-B",
        lower = c(1e-4, 1e-4),
        upper = c(5, 5)
    )

    c(
        lambda_hat = opt$par[1],
        mu_hat = opt$par[2],
        psi_hat = psi_fixed,
        loglik = -opt$value
    )
}


cl <- makeForkCluster(detectCores() - 1)
clusterSetRNGStream(cl, iseed = 19900330)
mle_ests <- pblapply(
    sim_list,
    function(sim) {
        estimates <- fit_one_mle(sim$tree, psi_fixed = 0.5, n0 = 1)
        truth <- sim$params
        data.frame(
            "lambda_hat" = estimates["lambda_hat"],
            "mu_hat" = estimates["mu_hat"],
            "psi_hat" = estimates["psi_hat"],
            "loglik" = estimates["loglik"],
            "true_lambda" = truth["lambda"],
            "true_mu" = truth["mu"],
            "true_psi" = truth["psi"],
            "ntips" = truth["ntips"]
        )
    },
    cl = cl
)
stopCluster(cl)


mle_mat <- do.call('rbind.data.frame', mle_ests)
row.names(mle_mat) <- 1:nrow(mle_mat)
colnames(mle_mat) <- c(
    "lambda_hat",
    "mu_hat",
    "psi_hat",
    "loglik",
    "true_lambda",
    "true_mu",
    "true_psi"
)
write.csv(mle_mat, "phylopomp_mle_lbdp.csv", row.names = TRUE)

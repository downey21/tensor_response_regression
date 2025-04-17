
# -*- coding: utf-8 -*-

rm(list = ls())

# https://github.com/lockEF/MultiwayRegression

# install.packages("MultiwayRegression")

library(MultiwayRegression)

# Array
a <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
b <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)

is.matrix(a)
is.array(a)

dim(a)

array(a, dim = c(2, 2))

# The contracted tensor product
MultiwayRegression::ctprod(a, b, K = 1)

a %*% b

# Tensor on tensor regression

# Fits a linear model to estimate one temsor from another tensor,
# under the restriction that the coefficient tensor has given PARAFAC rank R.

# loads simulated X: 100 x 15 x 20 and Y: 100 x 5 x 10
data(SimData, package = "MultiwayRegression")

head(X, n = 2)
dim(X)

apply(X, 2, mean)
apply(X, 3, mean)

head(Y, n = 2)
dim(Y)

# Fit rank 2 model with no regularization
Results <- MultiwayRegression::rrr(X, Y, R = 2, lambda = 0)

str(Results)
names(Results)

# sse: sum of squared residuals at each iteration
# sseR: value of the objective (sse+penalty) at each iteration

U_list <- Results$U
V_list <- Results$V
R <- 2

P <- sapply(U_list, nrow)
Q <- sapply(V_list, nrow)

# expand.grid
B_manual <- matrix(0, nrow = prod(P), ncol = prod(Q))
for (r in 1:R) {
    u_r <- apply(expand.grid(lapply(U_list, function(U) U[, r])), 1, prod)
    v_r <- apply(expand.grid(lapply(V_list, function(V) V[, r])), 1, prod)

    B_manual <- B_manual + outer(u_r, v_r)
}
B_manual <- array(B_manual, dim = c(P, Q))

Results$B == B_manual
all.equal(B_manual, Results$B, tolerance = 1e-6)

# kronecker
kronecker_vecs <- function(vec_list) {
    Reduce(kronecker, rev(vec_list))
}

B_manual <- matrix(0, nrow = prod(P), ncol = prod(Q))
for (r in 1:R) {
    u_r <- kronecker_vecs(lapply(U_list, function(U) U[, r]))
    v_r <- kronecker_vecs(lapply(V_list, function(V) V[, r]))

    B_manual <- B_manual + outer(u_r, v_r)
}

Results$B == B_manual
all.equal(B_manual, Results$B, tolerance = 1e-6)

# outer product
B_manual <- array(0, dim = c(P, Q))
for (r in 1:R) {
    u_vecs <- lapply(U_list, function(U) U[, r])
    v_vecs <- lapply(V_list, function(V) V[, r])

    rank1_tensor <- Reduce(outer, c(u_vecs, v_vecs))

    B_manual <- B_manual + rank1_tensor
}

Results$B == B_manual
max(Results$B - B_manual)
all.equal(B_manual, Results$B, tolerance = 1e-6)

# fitted tensor value
Y_pred <- MultiwayRegression::ctprod(X, Results$B, 2)

dim(Y_pred)

# Performs Bayesian posterior predictive inference
Results_bayesian <- MultiwayRegression::rrrBayes(X, Y, Inits = Results, R = 2, lambda = 0, Samples = 1000, X.new = X[1,,, drop = FALSE])

dim(X[1,,, drop = FALSE])

str(Results_bayesian)
dim(Results_bayesian)

# t-fold CV
t <- 5
set.seed(1)
folds <- sample(rep(1:t, length.out = dim(X)[1]))

R_vals <- 1:4
lambda_vals <- c(0, 0.1, 1, 10)

# for loop
# cv_errors <- matrix(
#     NA, nrow = length(R_vals), ncol = length(lambda_vals),
#     dimnames = list(paste0("R=", R_vals), paste0("lambda=", lambda_vals))
# )

# for (r_idx in seq_along(R_vals)) {
#     for (l_idx in seq_along(lambda_vals)) {

#         fold_errors <- numeric(t)

#         for (k in 1:t) {
#             test_idx <- which(folds == k)
#             train_idx <- setdiff(1:dim(X)[1], test_idx)

#             X_train <- X[train_idx,,,drop=FALSE]
#             Y_train <- Y[train_idx,,,drop=FALSE]
#             X_test <- X[test_idx,,,drop=FALSE]
#             Y_test <- Y[test_idx,,,drop=FALSE]

#             fit <- MultiwayRegression::rrr(X_train, Y_train, R = R_vals[r_idx], lambda = lambda_vals[l_idx])
#             Y_pred <- MultiwayRegression::ctprod(X_test, fit$B, 2)

#             fold_errors[k] <- sum((Y_test - Y_pred)^2) # Frobenius norm squared
#         }

#         cv_errors[r_idx, l_idx] <- mean(fold_errors)
#     }
# }

# mclapply
param_grid <- expand.grid(R = R_vals, lambda = lambda_vals)

cv_error_fn <- function(param) {
    r <- param$R
    lambda <- param$lambda

    fold_errors <- numeric(t)

    for (k in 1:t) {
        test_idx <- which(folds == k)
        train_idx <- setdiff(1:dim(X)[1], test_idx)

        X_train <- X[train_idx,,,drop=FALSE]
        Y_train <- Y[train_idx,,,drop=FALSE]
        X_test <- X[test_idx,,,drop=FALSE]
        Y_test <- Y[test_idx,,,drop=FALSE]

        fit <- MultiwayRegression::rrr(X_train, Y_train, R = r, lambda = lambda)
        Y_pred <- MultiwayRegression::ctprod(X_test, fit$B, 2)

        fold_errors[k] <- sum((Y_test - Y_pred)^2)
    }

    mean(fold_errors)
}

cores <- 4
cv_results <-
    parallel::mclapply(
        seq_len(nrow(param_grid)),
        function(i) cv_error_fn(param_grid[i, ]),
        mc.cores = cores,
        mc.set.seed = TRUE
    )

cv_errors <-
    matrix(
        unlist(cv_results),
        nrow = length(R_vals),
        ncol = length(lambda_vals),
        dimnames = list(paste0("R=", R_vals), paste0("lambda=", lambda_vals))
    )

print(cv_errors)
best_idx <- which(cv_errors == min(cv_errors), arr.ind = TRUE)
best_R <- R_vals[best_idx[1]]
best_lambda <- lambda_vals[best_idx[2]]

cat("Best R:", best_R, "\nBest lambda:", best_lambda, "\n")

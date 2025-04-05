
# -*- coding: utf-8 -*-

rm(list = ls())

library(rTensor)

source("/root/Project/tensor_response_regression/STORE/store.R")

EPS <- 1e-4

# setting of store:
# fix d1, d2, d3
# p = 1
# s (sparsity_level) in {0.3, 0.5}
# K (rank_K) in {2, 5}
# n (n_train) in {20, 100}
# n_rep = 200

set.seed(123)
n_train <- 20
n_test <- 100
p <- 3
dims <- c(10, 15, 20)
rank_K <- 2
sparsity_level <- 0.3
noise_sd <- 0.5
n_rep <- 2

methods <- c("mystor", "OLS", "Thresholded OLS", "Sparse OLS", "HOLRR")
pred_errors <- matrix(NA, nrow = n_rep, ncol = length(methods))
colnames(pred_errors) <- methods
estimation_errors <- matrix(NA, nrow = n_rep, ncol = length(methods))
colnames(estimation_errors) <- methods

sparsify_normalize <- function(vec, sparsity) {
    d <- length(vec)
    s0 <- ceiling(sparsity * d)
    idx <- sample(1:d, s0)
    out <- rep(0, d)
    out[idx] <- vec[idx]
    out / sqrt(sum(out^2))  # normalize
}

calc_rpe <- function(pred_list, true_list) {
    numerator <- sum(sapply(1:length(pred_list), function(i) {
        pred_tensor <- rTensor::as.tensor(pred_list[[i]])
        true_tensor <- rTensor::as.tensor(true_list[[i]])
        rTensor::fnorm(pred_tensor - true_tensor)^2
    }))
    denominator <- sum(sapply(1:length(true_list), function(i) {
        true_tensor <- rTensor::as.tensor(true_list[[i]])
        rTensor::fnorm(true_tensor)^2
    }))
    numerator / denominator
}

calc_estimation_error <- function(B_hat_est, B_true) {
    rTensor::fnorm(rTensor::as.tensor(B_hat_est) - rTensor::as.tensor(B_true))
}

for (iter in 1:n_rep) {
  
    X_train <- matrix(rnorm(n_train * p), nrow = n_train, ncol = p)
    X_test <- matrix(rnorm(n_test * p), nrow = n_test, ncol = p)
  
    B_true <- array(0, dim = c(dims, p))
    for (k in 1:rank_K) {
        betas <- lapply(dims, function(d) sparsify_normalize(rnorm(d), sparsity_level))
        beta4 <- sparsify_normalize(rnorm(p), sparsity_level)
        w_k <- rnorm(1)
        B_true <- B_true + w_k * array(Reduce(outer, c(betas, list(beta4))), dim = c(dims, p))
    }
    B_true_tensor <- rTensor::as.tensor(B_true)

    Y_train <- lapply(1:n_train, function(i) {
        noiseless <- array(rTensor::ttm(B_true_tensor, matrix(X_train[i, ], nrow = 1), m = length(dims) + 1)@data, dim = dims)
        noise <- array(rnorm(prod(dims), sd = noise_sd), dim = dims)
        noiseless + noise
    })

    Y_test <- lapply(1:n_test, function(i) {
        noiseless <- array(rTensor::ttm(B_true_tensor, matrix(X_test[i, ], nrow = 1), m = length(dims) + 1)@data, dim = dims)
        noise <- array(rnorm(prod(dims), sd = noise_sd), dim = dims)
        noiseless + noise
    })
  
    out_mystor <- mystor(Y_train, X_train, Para = rep(5, length(dims)), K = rank_K, stdmethod = "truncate", niter = 20)
    pred_mystor <- lapply(1:n_test, function(i) {
        array(rTensor::ttm(rTensor::as.tensor(out_mystor$B.hat), matrix(X_test[i, ], nrow = 1), m = length(dims) + 1)@data, dim = dims)
    })

    B_ols <- myols(Y_train, X_train)
    pred_ols <- lapply(1:n_test, function(i) {
        array(rTensor::ttm(rTensor::as.tensor(B_ols), matrix(X_test[i, ], nrow = 1), m = length(dims) + 1)@data, dim = dims)
    })

    B_truncateols <- mytruncateols(Y_train, X_train, gamma = 0.05)
    pred_truncateols <- lapply(1:n_test, function(i) {
        array(rTensor::ttm(rTensor::as.tensor(B_truncateols), matrix(X_test[i, ], nrow = 1), m = length(dims) + 1)@data, dim = dims)
    })

    B_sparseols <- mysparseols(Y_train, X_train, tuning = "BIC")
    pred_sparseols <- lapply(1:n_test, function(i) {
        array(rTensor::ttm(rTensor::as.tensor(B_sparseols), matrix(X_test[i, ], nrow = 1), m = length(dims) + 1)@data, dim = dims)
    })

    B_holrr <- myholrr(Y_train, X_train)
    pred_holrr <- lapply(1:n_test, function(i) {
        array(rTensor::ttm(rTensor::as.tensor(B_holrr), matrix(X_test[i, ], nrow = 1), m = length(dims) + 1)@data, dim = dims)
    })

    pred_errors[iter, "mystor"] <- calc_rpe(pred_mystor, Y_test)
    pred_errors[iter, "OLS"] <- calc_rpe(pred_ols, Y_test)
    pred_errors[iter, "Thresholded OLS"] <- calc_rpe(pred_truncateols, Y_test)
    pred_errors[iter, "Sparse OLS"] <- calc_rpe(pred_sparseols, Y_test)
    pred_errors[iter, "HOLRR"] <- calc_rpe(pred_holrr, Y_test)

    estimation_errors[iter, "mystor"] <- calc_estimation_error(out_mystor$B.hat, B_true)
    estimation_errors[iter, "OLS"] <- calc_estimation_error(B_ols, B_true)
    estimation_errors[iter, "Thresholded OLS"] <- calc_estimation_error(B_truncateols, B_true)
    estimation_errors[iter, "Sparse OLS"] <- calc_estimation_error(B_sparseols, B_true)
    estimation_errors[iter, "HOLRR"] <- calc_estimation_error(B_holrr, B_true)
}

pred_summary_table <- data.frame(
    Method = methods,
    Mean = unname(apply(pred_errors, 2, mean)),
    SD = unname(apply(pred_errors, 2, sd))
)

estimation_summary_table <- data.frame(
    Method = methods,
    Mean = unname(apply(estimation_errors, 2, mean)),
    SD = unname(apply(estimation_errors, 2, sd))
)

print(pred_summary_table)
print(estimation_summary_table)


# -*- coding: utf-8 -*-

rm(list = ls())

# rTensor R Package

# install.packages("rTensor")
# install.packages("MultiwayRegression")

library(rTensor)
library(MultiwayRegression)

set.seed(123)

# Tensor Objects

# From vector
vec <- runif(3)
vecT <- rTensor::as.tensor(vec)
vecT

# From matrix
mat <- matrix(runif(2 * 3), nrow = 2, ncol = 3)
matT <- rTensor::as.tensor(mat)
matT

# From array
arr <- array(runif(prod(c(2, 3, 4))), dim = c(2, 3, 4))
arrT <- rTensor::as.tensor(arr)
arrT

# Folding and Unfolding a Tensor

# Create a random 3D tensor
tnsr <- rTensor::rand_tensor(c(3, 4, 5))
tnsr

# (a) unfold: unfold tensor into matrix
# Choose row modes and column modes (e.g., mode 2 as row, modes 1 and 3 as columns)
unfolded_mat <- rTensor::unfold(tnsr, row_idx = 2, col_idx = c(1, 3))
dim(unfolded_mat)

# (b) fold: fold matrix back to tensor
folded_tnsr <- rTensor::fold(unfolded_mat, row_idx = 2, col_idx = c(1, 3), modes = c(3, 4, 5))
dim(folded_tnsr)

# Check if original and folded tensors are identical
identical(tnsr@data, folded_tnsr@data)

# (c) k_unfold: unfold along mode-k (common for Tucker)
k_unfolded <- rTensor::k_unfold(tnsr, m = 2)  # Unfold along mode 2
dim(k_unfolded)
identical(unfolded_mat@data, k_unfolded@data)

# (d) k_fold: fold matrix back after k_unfold
k_folded <- rTensor::k_fold(k_unfolded, m = 2, modes = c(3, 4, 5))
dim(k_folded)
identical(tnsr@data, k_folded@data)

# (e) vec: vectorize tensor
tnsr_vec <- rTensor::vec(tnsr)
length(tnsr_vec)
dim(tnsr_vec)

# ttm (tensor times matrix)
mat_6_4 <- matrix(runif(6 * 4), nrow = 6, ncol = 4)  # 6x4 matrix
tnsr_new <- rTensor::ttm(tnsr, mat_6_4, m = 2)

dim(tnsr@data)
dim(tnsr_new@data)

mat_7_5 <- matrix(runif(7 * 5), nrow = 7, ncol = 5)
tnsr_new_multi <- rTensor::ttl(tnsr, list_mat = list(mat_6_4, mat_7_5), ms = c(2, 3))
dim(tnsr_new_multi)

# modeSum
sum_mode2 <- rTensor::modeSum(tnsr, m = 2, drop = FALSE)
sum_mode2
dim(sum_mode2@data)

sum_mode2_manual <- apply(tnsr@data, MARGIN = c(1,3), sum)
sum_mode2_manual_tensor <- rTensor::as.tensor(array(sum_mode2_manual, dim = c(3, 1, 5)))
all.equal(sum_mode2@data, sum_mode2_manual_tensor@data)

# modeMean
mean_mode2 <- rTensor::modeMean(tnsr, m = 2, drop = FALSE)
mean_mode2
dim(mean_mode2@data)

mean_mode2_manual <- apply(tnsr@data, MARGIN = c(1,3), mean)
mean_mode2_manual_tensor <- rTensor::as.tensor(array(mean_mode2_manual, dim = c(3, 1, 5)))
all.equal(mean_mode2@data, mean_mode2_manual_tensor@data)

# innerProd
inner_AB <- rTensor::innerProd(tnsr, tnsr)

inner_AB_manual <- sum(tnsr@data * tnsr@data)
all.equal(inner_AB, inner_AB_manual)

# Frobenius Norm Calculation
rTensor::fnorm(tnsr)
sqrt(sum(tnsr@data^2))

sqrt(inner_AB)

# CP Decomposition
data_example <- rTensor::rand_tensor(c(4, 4, 4, 4))
cpD <- rTensor::cp(data_example, num_components = 3)

# Check if algorithm converged
cpD$conv

# Percentage of Frobenius norm explained
cpD$norm_percent

# Plot residual Frobenius norm during iterations
plot(cpD$all_resids, type = "b", main = "CP Decomposition Residuals", xlab = "Iteration", ylab = "Frobenius Norm Residual")

# check
lambdas <- cpD$lambdas
U_list <- cpD$U
num_components <- length(lambdas)
modes <- sapply(U_list, nrow)
num_modes <- length(modes)

# est_manual <- array(0, dim = modes)
# for (r in 1:num_components) {

#     outer_prod <- U_list[[1]][, r]

#     for (m in 2:num_modes) {
#     outer_prod <- outer(outer_prod, U_list[[m]][, r])
#     }

#     est_manual <- est_manual + lambdas[r] * array(outer_prod, dim = modes)
# }

est_manual <- array(0, dim = modes)
for (r in 1:num_components) {
    outer_prod <- Reduce(outer, lapply(U_list, function(U) U[, r]))
    est_manual <- est_manual + lambdas[r] * array(outer_prod, dim = modes)
}

sqrt(sum((cpD$est@data - est_manual)^2))
rTensor::norm_diff()


sqrt(sum((data_example@data - est_manual)^2))
cpD$fnorm_resid

# Tucker Decomposition
tuckerD <- rTensor::tucker(data_example, ranks = c(2, 2, 2, 2))

# Check if algorithm converged
tuckerD$conv

# Percentage of Frobenius norm explained
tuckerD$norm_percent

# Plot residual Frobenius norm during iterations
plot(tuckerD$all_resids, type = "b", main = "Tucker Decomposition Residuals", xlab = "Iteration", ylab = "Frobenius Norm Residual")

# check
Z <- tuckerD$Z # Core tensor
U_list <- tuckerD$U
modes <- sapply(U_list, nrow)
num_modes <- length(modes)

# est_manual <- Z
# for (m in 1:num_modes) {
#     est_manual <- rTensor::ttm(est_manual, U_list[[m]], m)
# }

est_manual <- rTensor::ttl(Z, U_list, ms = 1:num_modes)

sqrt(sum((tuckerD$est@data - est_manual@data)^2))

sqrt(sum((data_example@data - est_manual@data)^2))
tuckerD$fnorm_resid


tensor_contraction <- function(A, B, along = list(dimsA, dimsB)) {

    stopifnot(class(A) == "Tensor", class(B) == "Tensor")

    dimsA <- along[[1]]
    dimsB <- along[[2]]

    A_dim <- dim(A@data)
    B_dim <- dim(B@data)

    if (prod(A_dim[dimsA]) != prod(B_dim[dimsB])) {
        stop("Contracting dimensions must match in size.")
    }

    left_modes_A <- setdiff(seq_along(A_dim), dimsA)
    # A_mat <- array(A@data, dim = c(prod(A_dim[left_modes_A]), prod(A_dim[dimsA])))

    unfolded_A <- rTensor::unfold(A, row_idx = left_modes_A, col_idx = dimsA)
    A_mat <- unfolded_A@data

    right_modes_B <- setdiff(seq_along(B_dim), dimsB)
    # B_mat <- array(B@data, dim = c(prod(B_dim[dimsB]), prod(B_dim[right_modes_B])))

    unfolded_B <- rTensor::unfold(B, row_idx = dimsB, col_idx = right_modes_B)
    B_mat <- unfolded_B@data

    C_mat <- A_mat %*% B_mat

    final_dims <- c(A_dim[left_modes_A], B_dim[right_modes_B])
    C <- rTensor::as.tensor(array(C_mat, dim = final_dims))

    return(C)
}

A <- rTensor::rand_tensor(c(3, 4, 5))
B <- rTensor::rand_tensor(c(5, 4, 6))

C <- tensor_contraction(A, B, along = list(c(2, 3), c(1, 2)))

C@data
MultiwayRegression::ctprod(A@data, B@data, K = 2)

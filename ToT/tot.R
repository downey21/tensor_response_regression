
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

# Fit rank 2 model with no regularization
Results <- MultiwayRegression::rrr(X, Y, R = 2, lambda = 0)

# fitted tensor value
Y_pred <- MultiwayRegression::ctprod(X, Results$B, 2)


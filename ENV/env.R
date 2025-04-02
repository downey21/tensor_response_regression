
# -*- coding: utf-8 -*-

rm(list = ls())

# https://github.com/jingzzeng/TRES

# install.packages("TRES")

library(TRES)

## Load data "bat"
data("bat", package = "TRES")
x <- bat$x
y <- bat$y

## Fitting with OLS and 1D envelope method.
fit_ols <- TRR.fit(x, y, method="standard")

u <- TRRdim(x, y)$u
fit_1D <- TRR.fit(x, y, u = u, method="1D")
fit_1D_true <- TRR.fit(x, y, u = c(14, 14), method="1D")

fit_FG <- TRR.fit(x, y, u = u, method="FG")

coef(fit_1D)

fitted(fit_1D)

residuals(fit_1D)

predict(fit_1D, x)

summary(fit_1D)

summary_fit_1D <- summary(fit_1D)
names(summary_fit_1D)

summary_fit_1D$mse
summary_fit_1D$p_val
summary_fit_1D$se

#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

library("Matrix")

args <- commandArgs(TRUE)
X <- as.matrix(readMM(paste(args[1], "X.mtx", sep = "")))
method <- args[3]
standardize <- as.logical(args[4])

yeoJohnson <- function(x, lambda) {
    y <- numeric(length(x))
    nonnegative <- x >= 0

    if (abs(lambda) < 1e-12) {
        y[nonnegative] <- log1p(x[nonnegative])
    } else {
        y[nonnegative] <- ((x[nonnegative] + 1)^lambda - 1) / lambda
    }

    if (abs(lambda - 2) < 1e-12) {
        y[!nonnegative] <- -log1p(-x[!nonnegative])
    } else {
        y[!nonnegative] <-
            -(((1 - x[!nonnegative])^(2 - lambda) - 1) / (2 - lambda))
    }

    y
}

boxCox <- function(x, lambda) {
    if (abs(lambda) < 1e-12) {
        log(x)
    } else {
        (x^lambda - 1) / lambda
    }
}

transformColumn <- function(x, lambda, method) {
    if (method == "box-cox") boxCox(x, lambda) else yeoJohnson(x, lambda)
}

objective <- function(lambda, x, method) {
    y <- transformColumn(x, lambda, method)
    n <- length(x)
    variance <- sum((y - mean(y))^2) / n

    if (variance <= 0) {
        return(1e300)
    }

    logLikelihood <- -n / 2 * log(variance)
    if (method == "box-cox") {
        jacobian <- (lambda - 1) * sum(log(x))
    } else {
        jacobian <- (lambda - 1) * sum(sign(x) * log(abs(x) + 1))
    }

    -(logLikelihood + jacobian)
}

bracketMinimum <- function(x, method, initialLower = -2, initialUpper = 2) {
    goldenRatio <- 1.618034
    maxIterations <- 1000

    xa <- initialLower
    xb <- initialUpper
    fa <- objective(xa, x, method)
    fb <- objective(xb, x, method)

    if (fa < fb) {
        point <- xa
        xa <- xb
        xb <- point

        score <- fa
        fa <- fb
        fb <- score
    }

    xc <- xb + goldenRatio * (xb - xa)
    fc <- objective(xc, x, method)

    iteration <- 0
    while (fc < fb && iteration < maxIterations) {
        xa <- xb
        fa <- fb
        xb <- xc
        fb <- fc
        xc <- xb + goldenRatio * (xb - xa)
        fc <- objective(xc, x, method)
        iteration <- iteration + 1
    }

    validBracket <- (fb < fa && fb <= fc) || (fb <= fa && fb < fc)
    if (!validBracket) {
        stop("failed to bracket lambda minimum")
    }

    c(min(xa, xc), max(xa, xc))
}

lambdas <- matrix(1.0, nrow = 1, ncol = ncol(X))
Y <- matrix(0.0, nrow = nrow(X), ncol = ncol(X))

for (j in seq_len(ncol(X))) {
    x <- X[, j]
    if (max(x) != min(x)) {
        interval <- bracketMinimum(x, method)
        lambdas[1, j] <- optimize(
            objective,
            interval = interval,
            x = x,
            method = method,
            tol = 1.48e-8
        )$minimum
    }
    Y[, j] <- transformColumn(x, lambdas[1, j], method)
}

if (standardize) {
    means <- colMeans(Y)
    centered <- sweep(Y, 2, means)
    scales <- sqrt(colSums(centered^2) / nrow(Y))
    scales[scales == 0 | is.nan(scales)] <- 1
    Y <- sweep(centered, 2, scales, "/")
    state <- rbind(means, scales)
} else {
    state <- matrix(0.0, nrow = 2, ncol = ncol(X))
}

writeMM(as(Y, "CsparseMatrix"), paste(args[2], "Y", sep = ""))
writeMM(as(lambdas, "CsparseMatrix"), paste(args[2], "L", sep = ""))
writeMM(as(state, "CsparseMatrix"), paste(args[2], "S", sep = ""))

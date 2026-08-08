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
suppressPackageStartupMessages(library("recipes"))

args <- commandArgs(TRUE)
X <- as.matrix(readMM(paste(args[1], "X.mtx", sep = "")))
method <- args[3]
standardize <- as.logical(args[4])

colnames(X) <- paste0("V", seq_len(ncol(X)))
data <- as.data.frame(X)
transform <- recipe(~ ., data = data)

if (method == "box-cox") {
    transform <- step_BoxCox(
        transform,
        all_numeric(),
        limits = c(-20, 20),
        num_unique = 2
    )
} else {
    transform <- step_YeoJohnson(
        transform,
        all_numeric(),
        limits = c(-20, 20),
        num_unique = 2
    )
}

fitted <- prep(transform, training = data)
estimates <- tidy(fitted, number = 1)
lambdas <- matrix(1.0, nrow = 1, ncol = ncol(X))
lambdas[1, match(estimates$terms, colnames(X))] <- estimates$value
Y <- as.matrix(bake(fitted, new_data = data))

if (standardize) {
    observed <- colSums(!is.na(Y))
    means <- colMeans(Y, na.rm = TRUE)
    means[is.nan(means)] <- 0
    centered <- sweep(Y, 2, means)
    scales <- sqrt(colSums(centered^2, na.rm = TRUE) / observed)
    scales[observed == 0 | scales == 0 | is.nan(scales)] <- 1
    Y <- sweep(centered, 2, scales, "/")
    state <- rbind(means, scales)
} else {
    state <- matrix(0.0, nrow = 2, ncol = ncol(X))
}

writeMM(as(Y, "CsparseMatrix"), paste(args[2], "Y", sep = ""))
writeMM(as(lambdas, "CsparseMatrix"), paste(args[2], "L", sep = ""))
writeMM(as(state, "CsparseMatrix"), paste(args[2], "S", sep = ""))

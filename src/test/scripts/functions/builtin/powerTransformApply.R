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

# Read command line arguments passed from the Java test to the R script
args <- commandArgs(TRUE)

# Keep higher numeric output precision
options(digits = 22)

# Read test matrix X and lambda row matrix L from the input directory
X <- as.matrix(readMM(paste(args[1], "X.mtx", sep = "")))
lambdas <- as.matrix(readMM(paste(args[1], "L.mtx", sep = "")))

# Check every column has a matching lambda
if (ncol(X) != ncol(lambdas)) {
    stop("The number of lambdas must match the number of columns in X.")
}

# Apply the YJ transform with a given lambda to one vector
yeoJohnsonApply <- function(x, lambda) {
    y <- numeric(length(x))

    # Find positions of nonnegative and negative values separately to avoid computing the wrong branch
    nonnegativeIndex <- which(x >= 0)
    negativeIndex <- which(x < 0)

    # Handle the x >= 0 part
    if (length(nonnegativeIndex) > 0) {
        xPositive <- x[nonnegativeIndex]

        # Use the log form when lambda is close to 0
        if (abs(lambda) < 1e-12) {
            y[nonnegativeIndex] <- log(xPositive + 1)
        }
        else {
            y[nonnegativeIndex] <-
                ((xPositive + 1)^lambda - 1) / lambda
        }
    }

    # Handle the x < 0 part
    if (length(negativeIndex) > 0) {
        xNegative <- x[negativeIndex]

        # Use the log form when lambda is close to 2
        if (abs(lambda - 2) < 1e-12) {
            y[negativeIndex] <- -log(1 - xNegative)
        }
        else {
            y[negativeIndex] <-
                -(((1 - xNegative)^(2 - lambda) - 1) /
                    (2 - lambda))
        }
    }

    return(y)
}

# Create a result matrix with the same dimensions as the input matrix
Y <- matrix(
    0.0,
    nrow = nrow(X),
    ncol = ncol(X)
)

# Transform each column independently using its matching lambda
for (j in seq_len(ncol(X))) {
    lambda <- as.numeric(lambdas[1, j])
    Y[, j] <- yeoJohnsonApply(X[, j], lambda)
}

# Write the expected result computed by R to the expected output directory
writeMM(
    as(Y, "CsparseMatrix"),
    paste(args[2], "Y", sep = "")
)

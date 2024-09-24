#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------


args <- commandArgs(TRUE)

# Set options for numeric precision
options(digits=22)

# Load required libraries
library("Matrix")
library("matrixStats")

# Read matrix X from Matrix Market format files
X = as.matrix(readMM(paste(args[1], "X.mtx", sep="")))

type = as.integer(args[2])


# Perform the operations
if( type == 1 ) {
    #R = colMeans(X - colMeans(X))
    col_means <- matrix(colMeans(X), nrow = 1)
    # Subtract the row vector from each row of X
    centered_X <- sweep(X, 2, col_means, FUN = "-")
    # Calculate the column means of the centered matrix
    R <- colMeans(centered_X)
} else if ( type == 2) {
    # Compute column means and standard deviations
    col_means <- matrix(colMeans(X), nrow = 1)
    col_sds <- matrix(colSds(X), nrow = 1)
    # Center the matrix by subtracting column means
    centered_X <- sweep(X, 2, col_means, FUN = "-")
    # Scale the centered matrix by dividing by column standard deviations
    scaled_X <- sweep(centered_X, 2, col_sds, FUN = "/")
    # Compute the column means of the scaled matrix
    R <- colMeans(scaled_X)
}

writeMM(as(R, "CsparseMatrix"), paste(args[3], "R", sep=""))

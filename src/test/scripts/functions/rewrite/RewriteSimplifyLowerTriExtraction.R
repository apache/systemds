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

# Read command line arguments
args <- commandArgs(TRUE)

# Set options for numeric precision
options(digits=22)

# Load required libraries
library("Matrix")
library("matrixStats")

# Read matrix X from Matrix Market format files
X = as.matrix(readMM(paste(args[1], "X.mtx", sep="")))

# Create a column vector of ones with the same number of rows as X
ones_vector = matrix(1, nrow(X), 1)

# Create a diagonal matrix from the ones_vector
diag_matrix = diag(as.vector(ones_vector))

# Compute the column-wise cumulative sum (matching DML behavior)
cumsum_matrix = apply(diag_matrix, 2, cumsum)

# Perform the element-wise multiplication of X with the cumsum_matrix
R = X * cumsum_matrix

# Write the result R
writeMM(as(R, "CsparseMatrix"), paste(args[2], "R", sep=""))


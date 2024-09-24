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

# Define the vectors/matrices
X = matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9), nrow=9, ncol=1)
Y = matrix(1, nrow=9, ncol=1)
Z = matrix(1, nrow=9, ncol=1)

# Use table() to count unique combinations
R_table = table(X, Y, Z)
R = as.matrix(R_table)

writeMM(as(R, "CsparseMatrix"), paste(args[1], "R", sep=""))

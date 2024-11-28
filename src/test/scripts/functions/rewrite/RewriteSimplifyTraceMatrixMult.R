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


# Read matrices A, B, and C from Matrix Market format files
A = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
B = as.matrix(readMM(paste(args[1], "B.mtx", sep="")))

# Perform the matrix operation
R = sum(diag(A %*% B))
rA = A;
for(i in 1:nrow(rA)) {
  rA[,i] = rev(rA[,i])
}
R = R + sum(diag(rA))

# Write the result scalar R
write(R, paste(args[2], "R" ,sep=""))


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

# Read matrices A, B from Matrix Market format files
A = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
B = as.matrix(readMM(paste(args[1], "B.mtx", sep="")))
type = as.integer(args[2])

if( type == 1 ) {
      R = matrix(0, nrow(A), ncol(A)+ncol(B))
      R[,1]=A
      R[,2]=B
} else if( type == 2 ) {
      A_t = t(A)
      B_t = t(B)
      R = matrix(0, nrow(A_t)+nrow(B_t), ncol(A_t))
      R[1, ]=A_t
      R[2, ]=B_t
}

# Write the result R
writeMM(as(R, "CsparseMatrix"), paste(args[3], "R", sep=""))

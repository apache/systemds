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

# Read matrices U, V from Matrix Market format files
U = as.matrix(readMM(paste(args[1], "U.mtx", sep="")))
V = as.matrix(readMM(paste(args[1], "V.mtx", sep="")))
type = as.integer(args[2])
eps = 0.5

# Perform the operations
if( type == 1 ) {
    R = (eps + U%*%V)
} else if( type == 2 ) {
    R = (U%*%V - eps)
}


writeMM(as(R, "CsparseMatrix"), paste(args[3], "R", sep=""))

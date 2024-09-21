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

# Read matrices and type
U = as.matrix(readMM(paste(args[1], "U.mtx", sep="")))
V = as.matrix(readMM(paste(args[1], "V.mtx", sep="")))
W = as.matrix(readMM(paste(args[1], "W.mtx", sep="")))
type = as.integer(args[2])
c = 4

# Perform operations
if(type == 1){
    R = W * exp(U%*%t(V))
} else if(type == 2){
    R = W * abs(U%*%t(V))
} else if(type == 3){
    R = W * sin(U%*%t(V))
} else if(type == 4){
    R = (W*(U%*%t(V)))*2
} else if(type == 5){
    R = 2*(W*(U%*%t(V)))
} else if(type == 6){
    R = W * (c + U%*%t(V))
} else if(type == 7){
    R = W * (c - U%*%t(V))
} else if(type == 8){
    R = W * (c * (U%*%t(V)))
} else if(type == 9){
    R = W * (c / (U%*%t(V)))
} else if(type == 10){
    R = W * (U%*%t(V) + c)
} else if(type == 11){
    R = W * (U%*%t(V) - c)
} else if(type == 12){
    R = W * ((U%*%t(V)) * c)
} else if(type == 13){
    R = W * ((U%*%t(V)) / c)
}

#Write result matrix R
writeMM(as(R, "CsparseMatrix"), paste(args[3], "R", sep=""));


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
c = 4.0

# Perform operations
if(type == 1 || type == 14){
    R = W * exp(U%*%t(V))
} else if(type == 2 || type == 15){
    R = W * abs(U%*%t(V))
} else if(type == 3 || type == 16){
    R = W * sin(U%*%t(V))
} else if(type == 4 || type == 17){
    R = (W*(U%*%t(V)))*2
} else if(type == 5 || type == 18){
    R = 2*(W*(U%*%t(V)))
} else if(type == 6 || type == 19){
    R = W * (c + U%*%t(V))
} else if(type == 7 || type == 20){
    R = W * (c - U%*%t(V))
} else if(type == 8 || type == 21){
    R = W * (c * (U%*%t(V)))
} else if(type == 9 || type == 22){
    R = W * (c / (U%*%t(V)))
} else if(type == 10 || type == 23){
    R = W * (U%*%t(V) + c)
} else if(type == 11 || type == 24){
    R = W * (U%*%t(V) - c)
} else if(type == 12 || type == 25){
    R = W * ((U%*%t(V)) * c)
} else if(type == 13 || type == 26){
    R = W * ((U%*%t(V)) / c)
}

#Write result matrix R
writeMM(as(R, "CsparseMatrix"), paste(args[3], "R", sep=""));


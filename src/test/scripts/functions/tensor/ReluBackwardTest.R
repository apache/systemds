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
library("Matrix")
library("matrixStats") 
M=as.integer(args[1])
N=as.integer(args[2])

x=matrix(seq(1 - M, M*N - M), M, N, byrow=TRUE)
dout=matrix(seq(M*N, 1), M, N, byrow=TRUE)
output = (x > 0) * dout
writeMM(as(output,"CsparseMatrix"), paste(args[3], "B", sep=""))
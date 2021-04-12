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
args<-commandArgs(TRUE)
options(digits=22)
library("Matrix")
library("matrixStats")

X1 = as.matrix(readMM(paste(args[1], "X1.mtx", sep="")));
X2 = as.matrix(readMM(paste(args[1], "X2.mtx", sep="")));
X3 = as.matrix(readMM(paste(args[1], "X3.mtx", sep="")));
X4 = as.matrix(readMM(paste(args[1], "X4.mtx", sep="")));
X = rbind(X1, X2, X3, X4)
R = colProds(X)
writeMM(as(R, "CsparseMatrix"), paste(args[2], "S", sep=""));

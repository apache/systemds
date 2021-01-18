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

X = matrix(seq(1,3000), 60, 50, byrow=TRUE);
X[,5:45] = matrix(0, 60, 41);
# X[,2:4] = matrix(0, 60, 3);

Xmax = ((rowMaxs(X)%*%matrix(1,1,50))+0.5)
Xx = ((7+X)+(X-7)+(X^2<=7))
R = Xx/Xmax
print("X")
print.table(X[7:9,], digits=3, zero.print = ".")
print("R")
print.table(R[7:9,], digits=3, zero.print = ".")
print("Xx")
print.table(Xx[7:9,], digits=3, zero.print = ".")
print("Xmax")
print.table(Xmax[7:9,], digits=3, zero.print = ".")


writeMM(as(R, "CsparseMatrix"), paste(args[2], "S", sep="")); 

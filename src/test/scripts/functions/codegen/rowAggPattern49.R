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
# library("matrixStats")

W = matrix(seq(28,29), 1, 2)
J = matrix(0, 1, 8)
Z = cbind(J, W, J)
Y = matrix(0, 10, 18)
X = rbind(Z, Y, Y, Y, Y, Y, Y, Y, Y)
v = seq(1,81)
v1 = seq(20, 37)
W = matrix(seq(13,14), 1, 2)
J = matrix(0, 1, 8)
Z= cbind(J, W, J)
Y = matrix(0, 10, 18)
K = rbind(Z, Y, Y, Y, Y, Y, Y, Y, Y)

# S = (X < rowSums(X*K))
# S = X*rowMins(K)*X
# S = X*rowSums(K*v)*X
S = (X*v)/rowSums(X*v)
# S = abs((X*v)/rowSums(X*v))
# S = (X/v)+rowMeans(X-v)
# S = (X*v)+rowSums(X*v)
# S = (X*rowSums(X*v))/(X*v)

# S = X*rowSums(X*K)
# S = rowSums((X*v)/K)*v
# S = (K*v)/(rowSums(X*v))


writeMM(as(S, "CsparseMatrix"), paste(args[2], "S", sep=""));

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

X = readMM(paste(args[1], "X.mtx", sep=""));
v = seq(1, ncol(X));
w = seq(1, nrow(X));

R1 = t(X) %*% (X %*% v);
R2 = t(X) %*% (w * (X %*% v));
R = (R1 + R2) / 1e4;

writeMM(as(R,"CsparseMatrix"), paste(args[2], "R", sep=""));

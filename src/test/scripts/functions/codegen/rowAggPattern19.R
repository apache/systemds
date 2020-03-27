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

LT = matrix(1, 1500, 2);
Y = matrix(2, 1500, 2);
  
LT = LT - rowMaxs (LT) %*% matrix (1, 1, 2);
exp_LT = exp (LT);
R  = exp_LT / (rowSums (exp_LT) %*% matrix (1, 1, 2));
print(sum(Y * LT) + sum(log(rowSums(exp_LT))));

writeMM(as(R, "CsparseMatrix"), paste(args[2], "S", sep="")); 

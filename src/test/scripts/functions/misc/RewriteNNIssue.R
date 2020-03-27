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
library(Matrix)
library(matrixStats)

N = 2
C = 2
Hin = 3
Win = 4

X = matrix(cbind(seq(1,20),seq(1,20),seq(1,8)), nrow=2, ncol=24, byrow=TRUE)
gamma = matrix(c(1,2), byrow=TRUE, nrow=2, ncol=1)
beta = matrix(c(0,1), byrow=TRUE, nrow=2, ncol=1)
ema_mean = matrix(c(4,5), byrow=TRUE, nrow=2, ncol=1)
ema_var = matrix(c(2,3), byrow=TRUE, nrow=2, ncol=1)
mu = 0.95
epsilon = 1e-4

subgrp_means = matrix(colMeans(X), nrow=C, ncol=Hin*Win, byrow=TRUE)
subgrp_vars = matrix(colVars(X) * ((N-1)/N), nrow=C, ncol=Hin*Win, byrow=TRUE)
mean = rowMeans(subgrp_means)  # shape (C, 1)
var = rowMeans(subgrp_vars) + rowVars(subgrp_means)*(((Hin*Win)-1)/(Hin*Win))
ema_mean_upd = mu*ema_mean + (1-mu)*mean
ema_var_upd = mu*ema_var + (1-mu)*var

R = cbind(mean, var, ema_mean_upd, ema_var_upd)

writeMM(as(R, "CsparseMatrix"), paste(args[1], "R", sep=""));

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

X = readMM(paste(args[1], "X.mtx", sep=""));
W = readMM(paste(args[1], "W.mtx", sep=""));
H = readMM(paste(args[1], "H.mtx", sep=""));

k = as.integer(args[2]);
eps = as.double(args[3]);
max_iter = as.integer(args[4]);
iter = 1;

while( iter < max_iter ) {
   H = (H*(t(W)%*%(X/(W%*%H+eps)))) / (colSums(W)%*%matrix(1,1,ncol(H)));
   W = (W*((X/(W%*%H+eps))%*%t(H))) / (matrix(1,nrow(W),1)%*%t(rowSums(H)));
   obj = sum(W%*%H) - sum(X*log(W%*%H+eps));
   print(paste("obj=", obj))
   iter = iter + 1;
}

writeMM(as(W,"CsparseMatrix"), paste(args[5], "W", sep=""));
writeMM(as(H,"CsparseMatrix"), paste(args[5], "H", sep=""));

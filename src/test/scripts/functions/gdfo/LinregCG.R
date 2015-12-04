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
options(digits=22)
library("Matrix")

X = readMM(paste(args[1], "X.mtx", sep=""))
y = readMM(paste(args[1], "y.mtx", sep=""))

intercept = as.integer(args[2]);
eps = as.double(args[3]);
maxiter = as.double(args[4]);

if( intercept == 1 ){
   ones = matrix(1, nrow(X), 1); 
   X = cbind(X, ones);
}

r = -(t(X) %*% y);
p = -r;
norm_r2 = sum(r * r);
w = matrix(0, ncol(X), 1);

i = 0;
while(i < maxiter) {
	q = ((t(X) %*% (X %*% p)) + eps  * p);
	alpha = norm_r2 / ((t(p) %*% q)[1:1]);
	w = w + alpha * p;
	old_norm_r2 = norm_r2;
	r = r + alpha * q;
	norm_r2 = sum(r * r);
	beta = norm_r2 / old_norm_r2;
	p = -r + beta * p;
	i = i + 1;
}

writeMM(as(w,"CsparseMatrix"), paste(args[5], "w", sep=""))

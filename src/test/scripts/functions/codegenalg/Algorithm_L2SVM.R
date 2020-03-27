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
Y = readMM(paste(args[1], "Y.mtx", sep=""));
intercept = as.integer(args[2]);
epsilon = as.double(args[3]);
lambda = 0.001;
maxiterations = as.integer(args[4]);

check_min = min(Y)
check_max = max(Y)
num_min = sum(Y == check_min)
num_max = sum(Y == check_max)
if(num_min + num_max != nrow(Y)){ 
	print("please check Y, it should contain only 2 labels") 
}else{
	if(check_min != -1 | check_max != +1) 
		Y = 2/(check_max - check_min)*Y - (check_min + check_max)/(check_max - check_min)
}

dimensions = ncol(X)

if (intercept == 1) {
	ones  = matrix(1, rows=num_samples, cols=1)
	X = cbind(X, ones);
}

num_rows_in_w = dimensions
if(intercept == 1){
	num_rows_in_w = num_rows_in_w + 1
}
w = matrix(0, num_rows_in_w, 1)

g_old = t(X) %*% Y
s = g_old

Xw = matrix(0,nrow(X),1)
iter = 0
positive_label = check_max
negative_label = check_min

continue = TRUE
while(continue && iter < maxiterations){
	t = 0
	Xd = X %*% s
	wd = lambda * sum(w * s)
	dd = lambda * sum(s * s)
	continue1 = TRUE
	while(continue1){
		tmp_Xw = Xw + t*Xd
		out = 1 - Y * (tmp_Xw)
		sv = which(out > 0)
		g = wd + t*dd - sum(out[sv] * Y[sv] * Xd[sv])
		h = dd + sum(Xd[sv] * Xd[sv])
		t = t - g/h
		continue1 = (g*g/h >= 1e-10)
	}
	
	w = w + t*s
	Xw = Xw + t*Xd
		
	out = 1 - Y * (X %*% w)
	sv = which(out > 0)
	obj = 0.5 * sum(out[sv] * out[sv]) + lambda/2 * sum(w * w)
	g_new = t(X[sv,]) %*% (out[sv] * Y[sv]) - lambda * w
	
	print(paste("OBJ : ", obj))

	continue = (t*sum(s * g_old) >= epsilon*obj)
	
	be = sum(g_new * g_new)/sum(g_old * g_old)
	s = be * s + g_new
	g_old = g_new
	
	iter = iter + 1
}

extra_model_params = matrix(0, 4, 1)
extra_model_params[1,1] = positive_label
extra_model_params[2,1] = negative_label
extra_model_params[3,1] = intercept
extra_model_params[4,1] = dimensions

w = t(cbind(t(w), t(extra_model_params)))

writeMM(as(w,"CsparseMatrix"), paste(args[5], "w", sep=""));

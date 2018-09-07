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
max_iterations = as.integer(args[4]);


if(nrow(X) < 2)
	stop("Stopping due to invalid inputs: Not possible to learn a classifier without at least 2 rows")

lambda = 0.001
num_samples = nrow(X)
dimensions = ncol(X)
num_features = ncol(X)

min_y = min(Y)
num_classes = max(Y)
mod1 = Y %% 1
mod1_should_be_nrow = sum(abs(mod1==0))
	

if (intercept == 1) {
	ones  = matrix(1, num_samples, 1);
	X = append(X, ones);
}

num_rows_in_w = num_features
if(intercept == 1){
	num_rows_in_w = num_rows_in_w + 1
}
w = matrix(0, num_rows_in_w, num_classes)

debug_mat = matrix(-1, max_iterations, num_classes)
for(iter_class in 1:num_classes){		  
	Y_local = 2 * (Y == iter_class) - 1
	w_class = matrix(0, num_features, 1)
	if (intercept == 1) {
		zero_matrix = matrix(0, 1, 1);
		w_class = t(append(t(w_class), zero_matrix));
	}
 
	g_old = t(X) %*% Y_local
	s = g_old

	Xw = matrix(0, nrow(X), 1)
	iter = 0
	continue = 1
	while(continue == 1)  {
		# minimizing primal obj along direction s
 		step_sz = 0
 		Xd = X %*% s
 		wd = lambda * sum(w_class * s)
		dd = lambda * sum(s * s)
		continue1 = 1
		while(continue1 == 1){
 			tmp_Xw = Xw + step_sz*Xd
 			out = 1 - Y_local * (tmp_Xw)
 			sv = (out > 0)
 			out = out * sv
 			g = wd + step_sz*dd - sum(out * Y_local * Xd)
 			h = dd + sum(Xd * sv * Xd)
 			step_sz = step_sz - g/h
 			if (g*g/h < 0.0000000001){
			continue1 = 0
		}
	}
 
		#update weights
		w_class = w_class + step_sz*s
		Xw = Xw + step_sz*Xd
 
		out = 1 - Y_local * Xw
		sv = (out > 0)
		out = sv * out
		obj = 0.5 * sum(out * out) + lambda/2 * sum(w_class * w_class)
  		g_new = t(X) %*% (out * Y_local) - lambda * w_class

  		tmp = sum(s * g_old)
  
  		train_acc = sum( (Y_local*(X%*%w_class))>= 0)/num_samples*100
  		print(paste("For class " , iter_class , " iteration " , iter , " training accuracy: " , train_acc))
  		debug_mat[iter+1,iter_class] = obj	   
   
  		if((step_sz*tmp < epsilon*obj) | (iter >= max_iterations-1)){
   			continue = 0
  		}
 
  		#non-linear CG step
  		be = sum(g_new * g_new)/sum(g_old * g_old)
  		s = be * s + g_new
  		g_old = g_new

		if(sum(s^2) == 0){
	    	continue = 0
		}

  		iter = iter + 1
 	}

  w[,iter_class] = as.matrix(w_class)
}

extra_model_params = matrix(0, 2, ncol(w))
extra_model_params[1, 1] = intercept
extra_model_params[2, 1] = dimensions
w = t(cbind(t(w), t(extra_model_params)))

writeMM(as(w,"CsparseMatrix"), paste(args[5], "w", sep=""));

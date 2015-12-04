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
library(Matrix)

arima_css = function(w, X, p, P, q, Q, s, useJacobi){
  b = matrix(X[,2:ncol(X)], nrow(X), ncol(X)-1)%*%w

  R = matrix(0, nrow(X), nrow(X))
  if(q>0){
    for(i7 in 1:q){
      ma_ind_ns = P+p+i7
      err_ind_ns = i7
      ones_ns = rep(1, nrow(R)-err_ind_ns)
      d_ns = ones_ns * w[ma_ind_ns,1]
      R[(1+err_ind_ns):nrow(R),1:(ncol(R)-err_ind_ns)] = R[(1+err_ind_ns):nrow(R),1:(ncol(R)-err_ind_ns)] + diag(d_ns)
    }
  }
  if(Q>0){
    for(i8 in 1:Q){
      ma_ind_s = P+p+q+i8
      err_ind_s = s*i8
      ones_s = rep(1, nrow(R)-err_ind_s)
      d_s = ones_s * w[ma_ind_s,1]
      R[(1+err_ind_s):nrow(R),1:(ncol(R)-err_ind_s)] = R[(1+err_ind_s):nrow(R),1:(ncol(R)-err_ind_s)] + diag(d_s)
    }
  }

  max_iter = 100
  tol = 0.01

  y_hat = matrix(0, nrow(X), 1)
  iter = 0
  
  if(useJacobi == 1){
  	check = sum(ifelse(rowSums(abs(R)) >= 1, 1, 0))
  	if(check > 0){
  	  print("R is not diagonal dominant. Suggest switching to an exact solver.")
  	}
  	diff = tol+1.0
  	while(iter < max_iter & diff > tol){
  	  y_hat_new = matrix(b - R%*%y_hat, nrow(y_hat), 1)
  	  diff = sum((y_hat_new-y_hat)*(y_hat_new-y_hat))
  	  y_hat = y_hat_new
  	  iter = iter + 1
  	}
  }else{
  	ones = rep(1, nrow(X))
  	A = R + diag(ones)
  	Z = t(A)%*%A
  	y = t(A)%*%b
  	r = -y
  	p = -r
  	norm_r2 = sum(r*r)
  	while(iter < max_iter & norm_r2 > tol){
 	 	q = Z%*%p 
  		alpha = norm_r2 / sum(p*q)
  		y_hat = y_hat + alpha * p
  		old_norm_r2 = norm_r2
  		r = r + alpha * q
  		norm_r2 = sum(r * r)
  		beta = norm_r2 / old_norm_r2
  		p = -r + beta * p
  		iter = iter + 1
  	}
  }
  
  errs = X[,1] - y_hat
  obj = sum(errs*errs)

  return(obj)
}

#input col of time series data
X = readMM(paste(args[1], "col.mtx", sep=""))

max_func_invoc = as.integer(args[2])

#non-seasonal order
p = as.integer(args[3])
d = as.integer(args[4])
q = as.integer(args[5])

#seasonal order
P = as.integer(args[6])
D = as.integer(args[7])
Q = as.integer(args[8])

#length of the season
s = as.integer(args[9])

include_mean = as.integer(args[10])

useJacobi = as.integer(args[11])

num_rows = nrow(X)

if(num_rows <= d){
  print("non-seasonal differencing order should be larger than length of the time-series")
}

Y = matrix(X[,1], nrow(X), 1)
if(d>0){
  for(i in 1:d){
    n1 = nrow(Y)
    Y = matrix(Y[2:n1,] - Y[1:(n1-1),], n1-1, 1)
  }
}

num_rows = nrow(Y)
if(num_rows <= s*D){
  print("seasonal differencing order should be larger than number of observations divided by length of season")
}

if(D>0){
  for(i in 1:D){
    n1 = nrow(Y)
    Y = matrix(Y[(s+1):n1,] - Y[1:(n1-s),], n1-s, 1)
  }
}

n = nrow(Y)

max_ar_col = P+p
max_ma_col = Q+q
if(max_ar_col > max_ma_col){
  max_arma_col = max_ar_col
}else{
  max_arma_col = max_ma_col
}

mu = 0
if(include_mean == 1){
  mu = sum(Y)/nrow(Y)
  Y = Y - mu
}

totcols = 1+p+P+Q+q #target col (X), p-P cols, q-Q cols  

Z = matrix(0, n, totcols)
Z[,1] = Y #target col

if(p>0){
  for(i1 in 1:p){
    Z[(i1+1):n,1+i1] = Y[1:(n-i1),]
  }
}
if(P>0){
  for(i2 in 1:P){
    Z[(s*i2+1):n,1+p+i2] = Y[1:(n-s*i2),]
  }
}
if(q>0){
  for(i5 in 1:q){
    Z[(i5+1):n,1+P+p+i5] = Y[1:(n-i5),]
  }
}
if(Q>0){
  for(i6 in 1:Q){
     Z[(s*i6+1):n,1+P+p+q+i6] = Y[1:(n-s*i6),]
  }
}

simplex = matrix(0, totcols-1, totcols)
for(i in 2:ncol(simplex)){
  simplex[i-1,i] = 0.1
}

num_func_invoc = 0

objvals = matrix(0, 1, ncol(simplex))
for(i3 in 1:ncol(simplex)){
  objvals[1,i3] = arima_css(matrix(simplex[,i3], nrow(simplex), 1), Z, p, P, q, Q, s, useJacobi)
}
num_func_invoc = num_func_invoc + ncol(simplex)

tol = 1.5 * 10^(-8) * objvals[1,1]

continue = 1
while(continue == 1 & num_func_invoc <= max_func_invoc) {
  #print(paste(num_func_invoc, max_func_invoc))
  best_index = 1
  worst_index = 1
  for(i in 2:ncol(objvals)){
    this = objvals[1,i]
    that = objvals[1,best_index]
    if(that > this){
      best_index = i
    }
    that = objvals[1,worst_index]
    if(that < this){
      worst_index = i
    }
  }

  best_obj_val = objvals[1,best_index]
  worst_obj_val = objvals[1,worst_index]
  if(worst_obj_val <= best_obj_val + tol){
    continue = 0
  }

  print(paste("#Function calls::", num_func_invoc, "OBJ:", best_obj_val))

  c = (rowSums(simplex) - simplex[,worst_index])/(nrow(simplex))

  x_r = 2*c - simplex[,worst_index]
  obj_x_r = arima_css(matrix(x_r, nrow(simplex), 1), Z, p, P, q, Q, s, useJacobi)
  num_func_invoc = num_func_invoc + 1

  if(obj_x_r < best_obj_val){
    x_e = 2*x_r - c
    obj_x_e = arima_css(matrix(x_e, nrow(simplex), 1), Z, p, P, q, Q, s, useJacobi)
    num_func_invoc = num_func_invoc + 1

    if(obj_x_r <= obj_x_e){
      simplex[,worst_index] = x_r
      objvals[1,worst_index] = obj_x_r
    }else{
      simplex[,worst_index] = x_e
      objvals[1,worst_index] = obj_x_e
    }
  }else{
    if(obj_x_r < worst_obj_val){
      simplex[,worst_index] = x_r
      objvals[1,worst_index] = obj_x_r
    }

    x_c_in = (simplex[,worst_index] + c)/2
    obj_x_c_in = arima_css(matrix(x_c_in, nrow(simplex), 1), Z, p, P, q, Q, s, useJacobi)
    num_func_invoc = num_func_invoc + 1
    
    if(obj_x_c_in < objvals[1,worst_index]){
      simplex[,worst_index] = x_c_in
      objvals[1,worst_index] = obj_x_c_in
    }else{
      if(obj_x_r >= worst_obj_val){
        best_point = simplex[,best_index]

		for(i4 in 1:ncol(simplex)){
			if(i4 != best_index){
	          simplex[,i4] = (simplex[,i4] + best_point)/2
			  objvals[1,i4] = arima_css(matrix(simplex[,i4], nrow(simplex), 1), Z, p, P, q, Q, s, useJacobi)
			}
		}
		num_func_invoc = num_func_invoc + ncol(simplex) - 1
      }
    }
  }
}

best_point = matrix(simplex[,best_index], nrow(simplex), 1)
if(include_mean == 1){
  tmp = matrix(0, totcols, 1)
  tmp[1:nrow(best_point),1] = best_point
  tmp[nrow(tmp),1] = mu
  best_point = tmp
}

writeMM(as(best_point, "CsparseMatrix"), paste(args[12], "learnt.model", sep=""))

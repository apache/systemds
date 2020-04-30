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

#1. tanh function
func = function(X){
  Y = tanh(X)
  return(Y)
}

func1 = function(X) {
  Y = (exp(2*X) - 1)/(exp(2*X) + 1)
  Y_prime = 1 - Y^2
  return(Y_prime)
}

#2. feedForward

obj <- function(E){
  val = 0.5 * sum(E^2)
  return(val)
}


X = readMM(paste(args[1], "X.mtx", sep=""));
W1_rand = readMM(paste(args[1], "W1_rand.mtx", sep=""));
W2_rand = readMM(paste(args[1], "W2_rand.mtx", sep=""));
W3_rand = readMM(paste(args[1], "W3_rand.mtx", sep=""));
W4_rand = readMM(paste(args[1], "W4_rand.mtx", sep=""));
order_rand = readMM(paste(args[1], "order_rand.mtx", sep=""));

num_hidden1 = as.integer(args[2])    #$H1
num_hidden2 = as.integer(args[3])    #$H2
max_epochs = as.integer(args[4])     #$EPOCH
batch_size = as.integer(args[5])     #$BATCH

mu = 0.9 # momentum
step = 1e-5
decay = 0.95
hfile = " "
fmt = "text"
full_obj = FALSE

n = nrow(X)
m = ncol(X)

#randomly reordering rows
#permut = table(seq(from=1,to=n,by=1), order(runif(n, min=0, max=1)))
permut = table(seq(from=1,to=n,by=1), order(order_rand))
permut = as.data.frame.matrix(permut)
permut = data.matrix(permut)
X = (permut %*% X)
#z-transform, whitening operator is better
means = t(as.matrix(colSums(X)))/n
csx2 = t(as.matrix(colSums(X^2)))/n
stds = sqrt(csx2 - (means*means)*n/(n-1)) + 1e-17
X = (X - matrix(1, nrow(X),1) %*% means)/(matrix(1,nrow(X),1) %*% stds)

W1 = sqrt(6)/sqrt(m + num_hidden1) * W1_rand
b1 = matrix(0, num_hidden1, 1)

W2 = sqrt(6)/sqrt(num_hidden1 + num_hidden2) * W2_rand
b2 = matrix(0, num_hidden2, 2)

W3 = sqrt(6)/sqrt(num_hidden2 + num_hidden1) * W3_rand
b3 = matrix(0, num_hidden1, 1)

W4 = sqrt(6)/sqrt(num_hidden2 + m) * W4_rand
b4 = matrix(0, m, 1)

upd_W1 = matrix(0, nrow(W1), ncol(W1))
upd_b1 = matrix(0, nrow(b1), ncol(b1))
upd_W2 = matrix(0, nrow(W2), ncol(W2))
upd_b2 = matrix(0, nrow(b2), ncol(b2))
upd_W3 = matrix(0, nrow(W3), ncol(W3))
upd_b3 = matrix(0, nrow(b3), ncol(b3))
upd_W4 = matrix(0, nrow(W4), ncol(W4))
upd_b4 = matrix(0, nrow(b4), ncol(b4))

if( full_obj ){
    # nothing to do here
}

iter = 0
num_iters_per_epoch = ceiling(n / batch_size)
max_iterations = max_epochs * num_iters_per_epoch
# debug
# print("num_iters_per_epoch=" + num_iters_per_epoch + " max_iterations=" + max_iterations)
beg = 1
while( iter < max_iterations ) {
  end = beg + batch_size - 1
  if(end > n) end = n
  X_batch = X[beg:end,]

  # Notation:
  #  1    2         3   4         5   6         7     8          9
  # [H1, H1_prime, H2, H2_prime, H3, H3_prime, Yhat, Yhat_prime, E]
  # tmp_ff = feedForward(X_batch, W1, b1, W2, b2, W3, b3, W4, b4, X_batch)
  # H1 = tmp_ff[1]; H1_prime = tmp_ff[2]; H2 = tmp_ff[3]; H2_prime = tmp_ff[4];
  # H3 = tmp_ff[5]; H3_prime = tmp_ff[6]; Yhat = tmp_ff[7]; Yhat_prime = tmp_ff[8];
  # E = tmp_ff[9]
  # inputs: X, W1, b1, W2, b2, W3, b3, W4, b4, X_batch
  H1_in = t(W1 %*% t(X_batch) + b1 %*% matrix(1,ncol(b1),nrow(X_batch)))
  H1 = func(H1_in)
  H1_prime = func1(H1_in)

  H2_in = t(W2 %*% t(H1) + b2%*% matrix(1,ncol(b2),nrow(H1)))
  H2 = func(H2_in)
  H2_prime = func1(H2_in)

  H3_in = t(W3 %*% t(H2) + b3%*% matrix(1,ncol(b3),nrow(H2)))
  H3 = func(H3_in)
  H3_prime = func1(H3_in)

  Yhat_in = t(W4 %*% t(H3) + b4%*% matrix(1,ncol(b4),nrow(H3)))
  Yhat = func(Yhat_in)
  Yhat_prime = func1(Yhat_in)

  E = Yhat - X_batch

  # Notation:
  #  1        2           3      4        5         6        7      8
  # [W1_grad, b1_grad, W2_grad, b2_grad, W3_grad, b3_grad,W4_grad, b4_grad]
  # tmp_grad = grad(X_batch, H1, H1_prime, H2, H2_prime, H3, H3_prime, Yhat_prime, E, W1, W2, W3, W4)
  # W1_grad = tmp_grad[1]; b1_grad = tmp_grad[2]; W2_grad = tmp_grad[3]; b2_grad = tmp_grad[4];
  # W3_grad = tmp_grad[5]; b3_grad = tmp_grad[6]; W4_grad = tmp_grad[7]; b4_grad = tmp_grad[8];
  # grad function

  #backprop
  delta4 = E * Yhat_prime
  delta3 = H3_prime * (delta4 %*% W4)
  delta2 = H2_prime * (delta3 %*% W3)
  delta1 = H1_prime * (delta2 %*% W2)

  #compute gradients
  b4_grad = (colSums(delta4))
  b3_grad = (colSums(delta3))
  b2_grad = (colSums(delta2))
  b1_grad = (colSums(delta1))

  W4_grad = t(delta4) %*% H3
  W3_grad = t(delta3) %*% H2
  W2_grad = t(delta2) %*% H1
  W1_grad = t(delta1) %*% X_batch

  ob = obj(E)
  epochs = iter / num_iters_per_epoch
  # debug
  # print(table(epochs, ob), zero.print = "0")

  #update
  local_step = step / nrow(X_batch)
  upd_W1 = mu * upd_W1 - local_step * W1_grad
  upd_b1 = mu * upd_b1 - local_step * b1
  upd_W2 = mu * upd_W2 - local_step * W2_grad
  upd_b2 = mu * upd_b2 - local_step * b2
  upd_W3 = mu * upd_W3 - local_step * W3_grad
  upd_b3 = mu * upd_b3 - local_step * b3
  upd_W4 = mu * upd_W4 - local_step * W4_grad
  upd_b4 = mu * upd_b4 - local_step * b4
  W1 = W1 + upd_W1
  b1 = b1 + upd_b1
  W2 = W2 + upd_W2
  b2 = b2 + upd_b2
  W3 = W3 + upd_W3
  b3 = b3 + upd_b3
  W4 = W4 + upd_W4
  b4 = b4 + upd_b4

  iter = iter + 1
  if(end == n) beg = 1
  else beg = end + 1

  if(iter %% num_iters_per_epoch == 0) 
	  step = step * decay

  if(full_obj & iter %% num_iters_per_epoch == 0 ) {
    # Notation:
    # tmp_ff = feedForward(X, W1, b1, W2, b2, W3, b3, W4, b4, X)
    # full_H1 = tmp_ff[1]; full_H1_prime = tmp_ff[2]; full_H2 = tmp_ff[3]; full_H2_prime = tmp_ff[4];
    # full_H3 = tmp_ff[5]; full_H3_prime = tmp_ff[6]; full_Yhat = tmp_ff[7]; full_Yhat_prime = tmp_ff[8];
    # full_E = tmp_ff[9];
    # inputs: X, W1, b1, W2, b2, W3, b3, W4, b4, X
    H1_in = t(W1 %*% t(X) + b1 %*% matrix(1,ncol(b1),nrow(X)))

    full_H1 = func(H1_in)
    full_H1_prime = func1(H1_in)

    H2_in = t(W2 %*% t(H1) + b2%*% matrix(1,ncol(b2),nrow(H1)))
    full_H2 = func(H2_in)
    full_H2_prime = func1(H2_in)

    H3_in = t(W3 %*% t(H2) + b3%*% matrix(1,ncol(b3),nrow(H2)))
    full_H3 = func(H3_in)
    full_H3_prime = func1(H3_in)

    Yhat_in = t(W4 %*% t(H3) + b4%*% matrix(1,ncol(b4),nrow(H3)))
    full_Yhat = func(Yhat_in)
    full_Yhat_prime = func1(Yhat_in)
    full_E = full_Yhat - X

    full_o = obj(full_E)
    epochs = iter %/% num_iters_per_epoch
    # debug
    # print(table(epochs, full_o, deparse.level=2), zero.print=".")
    # print("EPOCHS=" + epochs + " iter=" + iter + " OBJ (FULL DATA)=" + full_o)
  }
}

#debug
#print.table(W1, digits=3)
writeMM(as(W1,"CsparseMatrix"), paste(args[6], "W1", sep=""));
writeMM(as(b1,"CsparseMatrix"), paste(args[6], "b1", sep=""));
writeMM(as(W2,"CsparseMatrix"), paste(args[6], "W2", sep=""));
writeMM(as(b2,"CsparseMatrix"), paste(args[6], "b2", sep=""));
writeMM(as(W3,"CsparseMatrix"), paste(args[6], "W3", sep=""));
writeMM(as(b3,"CsparseMatrix"), paste(args[6], "b3", sep=""));
writeMM(as(W4,"CsparseMatrix"), paste(args[6], "W4", sep=""));
writeMM(as(b4,"CsparseMatrix"), paste(args[6], "b4", sep=""));

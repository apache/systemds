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

# JUnit test class: dml.test.integration.applications.LinearLogReg.java
# command line invocation assuming $LLR_HOME is set to the home of the R script
# Rscript $LLR_HOME/LinearLogReg.R $LLR_HOME/in/ $LLR_HOME/expected/

args <- commandArgs(TRUE)

library("Matrix")
#library("batch")
# Usage:  /home/vikas/R-2.10.1/bin/R --vanilla --args Xfile X yfile y Cval 2 tol 0.01 maxiter 100 < linearLogReg.r

# Solves Linear Logistic Regression using Trust Region methods. 
# Can be adapted for L2-SVMs and more general unconstrained optimization problems also
# setup optimization parameters (See: Trust Region Newton Method for Logistic Regression, Lin, Weng and Keerthi, JMLR 9 (2008) 627-650)

options(warn=-1)

C = 2; 
tol = 0.001
maxiter = 3
maxinneriter = 3

eta0 = 0.0001
eta1 = 0.25
eta2 = 0.75
sigma1 = 0.25
sigma2 = 0.5
sigma3 = 4.0
psi = 0.1 

# read (training and test) data files -- should be in matrix market format. see data.mtx 
X = readMM(paste(args[1], "X.mtx", sep=""));
Xt = readMM(paste(args[1], "Xt.mtx", sep=""));

N = nrow(X)
D = ncol(X)
Nt = nrow(Xt)

# read (training and test) labels
y = readMM(paste(args[1], "y.mtx", sep=""));
yt = readMM(paste(args[1], "yt.mtx", sep=""));

# initialize w
w = matrix(0,D,1)
o = X %*% w
logistic = 1.0/(1.0 + exp(-y*o))

# VS : change
obj = 0.5 * t(w) %*% w + C*sum(-log(logistic))
grad = w + C*t(X) %*% ((logistic - 1)*y)
logisticD = logistic*(1-logistic)
delta = sqrt(sum(grad*grad))

# number of iterations
iter = 0

# starting point for CG
zeros_D = matrix(0,D,1)
# VS: change
zeros_N = matrix(0,N,1)

# boolean for convergence check

converge = (delta < tol)
norm_r2 = sum(grad*grad)
gnorm = sqrt(norm_r2)
# VS: change
norm_grad = sqrt(norm_r2)
norm_grad_initial = norm_grad

while(!converge) {
 	
	norm_grad = sqrt(sum(grad*grad))

	print("next iteration..")
	print(paste("Iterations : ",iter, "Objective : ", obj[1,1],  "Gradient Norm : ", norm_grad))
    	 
	# SOLVE TRUST REGION SUB-PROBLEM
	s = zeros_D
	os = zeros_N
	r = -grad
	d = r
	innerconverge = (sqrt(sum(r*r)) <= psi*norm_grad)
	inneriter = 0; 	
	while(!innerconverge) {
		inneriter = inneriter + 1
		norm_r2 = sum(r*r)
		od = X %*% d
		Hd = d + C*(t(X) %*% (logisticD*od))
		alpha_deno = t(d) %*% Hd 
		alpha = norm_r2/alpha_deno
		
		s = s + alpha[1,1]*d
		os = os + alpha[1,1]*od

		sts = t(s) %*% s
		delta2 = delta*delta 
		
		if (sts[1,1] > delta2) {
			# VS: change 
			print("cg reaches trust region boundary")
			# VS: change
			s = s - alpha[1,1]*d
			os = os - alpha[1,1]*od
			std = t(s) %*% d
			dtd = t(d) %*% d
			# VS:change
			sts = t(s) %*% s
			rad = sqrt(std*std + dtd*(delta2 - sts))
			if(std[1,1]>=0) {
				tau = (delta2 - sts)/(std + rad)
			} 
			else {
				tau = (rad - std)/dtd
			}	 
			s = s + tau[1,1]*d
			os = os + tau[1,1]*od
			r = r - tau[1,1]*Hd
			break
		}
		r = r - alpha[1,1]*Hd
		old_norm_r2 = norm_r2 
		norm_r2 = sum(r*r)
		beta = norm_r2/old_norm_r2
		d = r + beta*d
		innerconverge = (sqrt(norm_r2) <= psi * norm_grad) | (inneriter > maxinneriter) # innerconverge = (sqrt(norm_r2) <= psi*norm_grad)
	}
	
	print(paste("Inner CG Iteration = ", inneriter))
	# END TRUST REGION SUB-PROBLEM
	# compute rho, update w, obtain delta
	gs = t(s) %*% grad
	qk = -0.5*(gs - (t(s) %*% r))
	
	wnew = w + s
	# VS Change X %*% wnew removed	
	onew = o + os 
	# VS: change
	logisticnew = 1.0/(1.0 + exp(-y*onew))
	objnew = 0.5 * t(wnew) %*% wnew + C*sum(-log(logisticnew))

	# VS: change
	actred = (obj - objnew)	
	rho = actred/qk

	print(paste("Actual :", actred[1,1], "Predicted :", qk[1,1]))

	rho = rho[1,1]
	snorm = sqrt(sum(s*s))

	if(iter==0) {
		delta = min(delta, snorm)
	}
	if (objnew[1,1] - obj[1,1] - gs[1,1] <= 0) {
		alpha = sigma3;
	}
	else {
		alpha = max(sigma1, -0.5*gs[1,1]/(objnew[1,1] - obj[1,1] - gs[1,1]))
	}



	if (rho > eta0) {
	
		w = wnew
		o = onew
		grad = w + C*t(X) %*% ((logisticnew - 1)*y)
		# VS: change
		norm_grad = sqrt(sum(grad*grad))
		logisticD = logisticnew*(1-logisticnew)
		obj = objnew

	}
	
	if (rho < eta0)
		{delta = min(max(alpha, sigma1)*snorm, sigma2*delta)}
	else if (rho < eta1)
		{delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta))}
	else if (rho < eta2)
		{delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta))}
	else
		{delta = max(delta, min(alpha*snorm, sigma3*delta))}

	ot = Xt %*% w
	correct = sum((yt*ot)>0)
	iter = iter + 1
	converge = (norm_grad < tol*norm_grad_initial) | (iter>maxiter)
	
	print(paste("Delta :", delta))
	print(paste("Accuracy=", correct*100/Nt))
	print(paste("OuterIter=", iter))
	print(paste("Converge=", converge))
}

writeMM(as(w,"CsparseMatrix"), paste(args[2],"w", sep=""), format = "text")



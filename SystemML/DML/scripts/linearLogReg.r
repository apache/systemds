# DOUG TEST CASE
library("Matrix")
library("batch")
# Usage:  /home/vikas/R-2.10.1/bin/R --vanilla --args Xfile X yfile y C 2 tol 1e-6 maxiter 100 < linearLogReg.r
parseCommandArgs()

# Solves Linear Logistic Regression using Trust Region methods. 
# Can be adapted for L2-SVMs and more general unconstrained optimization problems also
# setup optimization parameters (See: Trust Region Newton Method for Logistic Regression, Lin, Weng and Keerthi, JMLR 9 (2008) 627-650)
eta0 = 0.0001
eta1 = 0.25
eta2 = 0.75
sigma1 = 0.25
sigma2 = 0.5
sigma3 = 4.0
psi = 0.1 
maxinneriter = 1000 

# read A file - should be in matrix market format. see data.mtx 
X = readMM(file=Xfile)
C = Cval
N = nrow(X)
D = ncol(X)

# read b file
y = as.matrix(scan(file=yfile),N,1)

# initialize w
w = matrix(0,D,1)
o = X %*% w
logistic = 1.0/(1.0 + exp(-y*o))

obj = 0.5 * t(w) %*% w + C*sum(logistic)
grad = w + C*t(X) %*% ((logistic - 1)*y)
logisticD = logistic*(1-logistic)
delta = sqrt(sum(grad*grad))

# number of iterations
iter = 0

# starting point for CG
zeros_D = matrix(0,D,1)

# boolean for convergence check

converge = (delta < tol)

norm_r2 = sum(grad*grad)

while(!converge) {
	
	print(paste("Iterations : ",iter, "Gradient Norm : ", sqrt(norm_r2)))
    	
	norm_grad = sqrt(sum(grad*grad))
	# SOLVE TRUST REGION SUB-PROBLEM
	s = zeros_D
	r = -grad
	d = r
	innerconverge = (sqrt(sum(r*r)) <= psi*norm_grad)
	
	while(!innerconverge) {
		norm_r2 = sum(r*r)
		Hd = d + C*(t(X) %*% (logisticD*(X %*% d)))
		alpha_deno = t(d) %*% Hd 
		alpha = norm_r2/alpha_deno
		s = s + alpha[1,1]*d
		sts = t(s) %*% s
		delta2 = delta*delta 
		if (sts[1,1] > delta2) {
			std = t(s) %*% d
			dtd = t(d) %*% d
			rad = sqrt(std*std + dtd*(delta2 - sts))
			if(std>=0) {
				tau = (delta2 - sts)/(std + rad)
			} 
			else {
				tau = (rad - std)/dtd
				}	 
			s = s + tau*d
			r = r - tau*Hd
			break
		}
		r = r - alpha[1,1]*Hd
		old_norm_r2 = norm_r2 
		norm_r2 = sum(r*r)
		beta = norm_r2/old_norm_r2
		d = r + beta*d
		innerconverge = (sqrt(norm_r2) <= psi*norm_grad)
	}

	# END TRUST REGION SUB-PROBLEM
	# compute rho, update w, obtain delta
	qk = -0.5*(t(s) %*% (grad - r))
	
	wnew = w + s	
	onew = X %*% wnew
	logisticnew = 1.0/(1.0 + exp(-y*o))
	objnew = 0.5 * t(wnew) %*% wnew + C*sum(logisticnew)
	
	rho = (objnew - obj)/qk
	rho = rho[1,1]
	snorm = sqrt(sum(s*s))

	if (rho > eta0) {
	
		w = wnew
		o = onew
		grad = w + C*t(X) %*% ((logisticnew - 1)*y)
		logisticD = logisticnew*(1-logisticnew)
		}
	
	if (rho < eta0)
			{delta = min(max(alpha, sigma1)*snorm, sigma2*delta)}
		else if (rho < eta1)
			{delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta))}
		else if (rho < eta2)
			{delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta))}
		else
			{delta = max(delta, min(alpha*snorm, sigma3*delta))}
		 		
	iter = iter + 1

	converge = (norm_r2 < tol*tol) | (iter>maxiter)
}


library("Matrix")
library("batch")
# Usage:  /home/vikas/R-2.10.1/bin/R --vanilla --args Xfile X yfile y lambda 1e-6 tol 1e-6 maxiter 100 < linearCG.r
parseCommandArgs()

# read A file - should be in matrix market format. see data.mtx
 
X = readMM(file=Xfile)

N = nrow(X)
D = ncol(X)

# read b file
y = as.matrix(scan(file=yfile),N,1)
b = t(X) %*% y

# initialize the solution x
x = matrix(0,D,1)
 
# residual
r = ((t(X) %*% (X %*%x)) + lambda*x) - b

# initial direction
p = -r

# number of iterations
iter = 0

# boolean for convergence check

norm_r2 = sum(r*r)

converge = (norm_r2 < tol*tol)

while(!converge) {
	
	print(paste("Iterations : ",iter, "Gradient Norm : ", sqrt(norm_r2)))
    	
        q = ((t(X) %*% (X %*% p)) + lambda*p)

	alpha = (norm_r2)/(t(p) %*% q)
	
	x = x + alpha[1,1] * p
	
	old_norm_r2 = norm_r2

	r = r + (alpha[1,1] * q)

	norm_r2 =  sum(r*r) 

	beta = norm_r2/old_norm_r2

	p = -r + (beta * p)
	
	iter = iter + 1

	converge = (norm_r2 < tol*tol) | (iter>maxiter)
}

##r = (t(X) %*% (X %*%x) + lambda*x)- b

##print(paste("Iterations : ",iter, "Final Gradient Norm Squared :", t(r) %*% r))

print(x)
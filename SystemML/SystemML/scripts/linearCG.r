library("Matrix")
library("batch")
# Usage:  /home/vikas/R-2.10.1/bin/R --vanilla --args Afile A bfile b tol 1e-6 maxiter 100 < linearCG.r
parseCommandArgs()

# read A file - should be in matrix market format. see data.mtx
 
A = readMM(file=Afile)
N = nrow(A)
D = ncol(A)

# read b file
b = as.matrix(scan(file=bfile),N,1)

# initialize the solution x
x = matrix(0,D,1)
 
# residual
r = A%*%x - b

# initial direction
p = -r

# number of iterations
iter = 0

# boolean for convergence check

norm_r2 = norm(r, 'f')^2

converge = (norm_r2 < tol*tol)

while(!converge) {
	
	print(paste("Iterations : ",iter, "Gradient Norm : ", norm_r2))
    	
	alpha = (norm_r2)/(t(p) %*% (A %*% p))
	
	x = x + alpha[1,1] * p
	
	old_norm_r2 = norm_r2

	r = r + (alpha[1,1] * (A %*% p))

	norm_r2 =  norm(r, 'f')^2 

	beta = norm_r2/old_norm_r2

	p = -r + (beta * p)
	
	iter = iter + 1

	converge = (norm_r2 < tol*tol) | (iter>maxiter)
}


print(paste("Iterations : ",iter, "Final Gradient Norm Squared :", norm(A %*%x - b)^2))
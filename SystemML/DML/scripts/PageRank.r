library("Matrix")
library("batch")
# Usage:  /home/vikas/R-2.10.1/bin/R --vanilla --args graphfile G tol 1e-6 maxiter 100 alpha 0.85 < PageRank.r
parseCommandArgs()

# read graph file - should be in matrix market format. see data.mtx
  
G = readMM(file=graphfile)

N = nrow(G)
D = ncol(G)

# note: assuming N=D and also assuming no dangling node (i.e., no row is all-zero)

# row-normalize G
outdegree = rowSums(G)

G = sparseMatrix(i=1:N, j=1:N, x = 1.0/outdegree) %*% G

# initial value of pagerank  
p = matrix(1,N,1)/N

e = matrix(1,N,1)

# teleportation vector - uniform
u = matrix(1,N,1)/N

 
# number of iterations
iter = 0

# boolean for convergence check

converge = FALSE

delta = sum(p*p)

while(!converge) {
	
	print(paste("Iterations : ",iter, "PR diff : ", sqrt(delta)))
    	
        old_p = p

        p = alpha*(G %*% p) + (1-alpha)*(e %*% (t(u) %*% p))
	
	iter = iter + 1
        
        delta  = sum((p-old_p)*(p-old_p))

	converge = ( delta < tol*tol) | (iter>maxiter)
}
 

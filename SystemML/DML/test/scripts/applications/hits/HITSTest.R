library("Matrix")
#library("batch")
# Usage:  R --vanilla --args graphfile G tol 1e-8 maxiter 100 < HITS.R
#parseCommandArgs()

maxiter = $$maxiter$$
tol = $$tol$$

G = readMM(file="$$indir$$graph.mtx")
authorities = readMM(file="$$indir$$init_authorities.mtx")
hubs = authorities

#N = nrow(G)
#D = ncol(G)

 
# HITS = power iterations to compute leading left/right singular vectors
 
#authorities = matrix(1.0/N,N,1)
#hubs = matrix(1.0/N,N,1)

converge = FALSE
iter = 0

while(!converge) {
	
	hubs_old = hubs
	hubs = G %*% authorities

	authorities_old = authorities
	authorities = t(G) %*% hubs

	hubs = hubs/max(hubs)
	authorities = authorities/max(authorities)

	delta_hubs = sum((hubs - hubs_old)^2)
	delta_authorities = sum((authorities - authorities_old)^2)

	converge = ((abs(delta_hubs) < tol) & (abs(delta_authorities) < tol) | (iter>maxiter))
	
	iter = iter + 1
	print(paste("Iterations :", iter, " delta_hubs :", delta_hubs, " delta_authorities :", delta_authorities))
}

writeMM(as(hubs,"CsparseMatrix"),"$$Routdir$$hubs",format="text")
writeMM(as(authorities,"CsparseMatrix"),"$$Routdir$$authorities",format="text")

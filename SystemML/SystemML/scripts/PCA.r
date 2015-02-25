library("Matrix")
library("batch")
# Usage:  /home/vikas/R-2.10.1/bin/R --vanilla --args data X ncomponents 100 < PCA.r
parseCommandArgs()

# read A file - should be in matrix market format. see data.mtx
   
#X = readMM(file=data)

set.seed(123)
N = 10
D = 5
components = 3


# computations on hadoop
X <- matrix(c(runif(N*D)),N,D);
mu = colSums(X)/N;
C = (t(X) %*% X)/(N-1) -  (N/(N-1))*mu %*% t(mu)
C
#call to jlapack
ev <- eigen(C)
 
evec = ev$vec
eval = ev$val

evec = evec[,1:components]
eval = eval[1:components]
print(evec)
print(eval)
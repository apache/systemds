#read A from file
n <- 3
m <- 3
#a <- scan("..\\data\\R-fileA", list(0))
#A <- t(array (a[[1]], dim=c(m,n)))

#generate A randomly
n <- 10000
m <- 10000
nel <- n * m
bmax <- 1000
at <- runif(nel, 1, bmax)
A <- array(at,dim=c(n,m))

y <- array ( 1 ,dim=m);
b <- array ( 1 ,dim=n);
c <- array ( 1 ,dim=m);


Amax = max(A);
eps = 3.9;
mu = 1/eps * log ((m*Amax)/eps);
alpha=eps/4;
beta = alpha / (10*mu);
delta = alpha / (10*mu*n*Amax);


i = 1;
max_iter = 30000;
potential=1000;
prev_potential=0;


while ( (eps > 0.001) & (i <= max_iter) ) {
    if (abs(potential - prev_potential) < 10^-7) {
        eps = eps / 1.1;
        mu = 1/eps * log ((m*Amax)/eps);
        alpha=eps/4;
        beta = alpha / (10*mu);
        delta = alpha / (10*mu*n*Amax);
    }
    # Step 1
    oy = A %*% y;
    x = exp(mu*(1-oy));

        
    # Step 2
    ox = t(A) %*% x;
    
    v1 = ox >= 1+alpha;
    v2 = ox <= 1-alpha;
    v3 = (v1 + v2) == 0;
    v1 <- drop(v1)
    v2 <- drop(v2)
    v3 <- drop(v3)

    y  = v1*pmax((1+beta)*y, array(delta,dim=m)) + pmin(v2*y*(1-beta)) + v3 * y 

    cat ("Iter: ",i, "Obj: ",sum(y),"eps ",eps,"\n")

    prev_potential = potential;
    potential = sum(y) + sum(x)/mu;

    
    i = i+1;

}
write(y, file="y.result");

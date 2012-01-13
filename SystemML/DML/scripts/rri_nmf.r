library("Matrix")
library("batch")
parseCommandArgs()

# read V file
dat <- scan('GATORADE_HYDRATION2.spmat',list("character", "integer","integer","integer"),nmax<-1);
tab <- read.table(vfile);
N <- as.integer(dat[[2]])
D <- as.integer(dat[[3]])
NNZ <- as.integer(dat[[4]])
   
 
V <- sparseMatrix(i = tab[,1],
                  j = tab[,2],
                  x = tab[,3],
		   dims = c(N,D));


# initialize W, H
W <- matrix(c(runif(N*K)),N,K);
H <- matrix(c(runif(K*D)),K,D);
 
alpha <- sum(diag(t(W) %*% V %*% t(H)))/sum(diag((H %*% t(H)) %*% (t(W)%*%W)));
W <- sqrt(alpha)*W;
H <- sqrt(alpha)*H;
 
obj <- Inf;

normV <- sum(V*V);

iter = 0.0;

terminate <- TRUE;


while(terminate) {
    
    if (iter >= maxiter) {
        terminate <- FALSE;
	next;
    }
    
    obj2 <-0.0;
    obj3a <-0.0;
    obj3b <-0.0;
    obj4  <- 0.0;

    for (r in 1:K) {
       
       # what we are optimizing in this round 
       h <- H[r,];
       Hh <- as.vector(H %*% h);

       Hh[r]<-0;
       Vh = V %*% h;
       g = Vh - W %*% Hh;
       Rh <- pmax(as.vector(g),0);
 
       w <- Rh/((t(h) %*% h + epsilon))[1,1];
       W[,r] <- w;
       
       obj2  <- obj2 + t(w) %*% Vh;
       obj3a <- obj3a + t(w) %*% W[ ,1:r-1] %*% Hh[1:r-1];
       hh <- t(h) %*% h;
       ww <- t(w) %*% w;	 
       obj3b <- obj3b + (hh)*(ww);
       obj4 <- obj4 + (hh) + (ww);
    }
     
    ##%%%%%%%%%%%%%%%%% CONVERGENCE CHECK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    prev_obj = obj;
    
    obj <- normV - 2*obj2 + (2*obj3a +obj3b) + epsilon*obj4; ## last term uses definition of trace + symmetry
    
    iter = iter + 1.0;
    
    print(paste("Iterations=",iter, "Objective=", obj[1,1]));
    check = (prev_obj < obj);

    if (check[1,1]) {
        	print("obj function increased -- something wrong.\n");
    }
    check = (abs(prev_obj-obj) < prev_obj*tol) ;
 
    if (check[1,1]) {
        terminate <- TRUE;
     }
    
    ##%%%%%%%%%%%%%%%%%%%%%%%%%% CONVERGENCE CHECK END %%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
   for (r in 1:K) {
       
       ## update equations for h 
   
       w <- W[,r];
       Ww <- as.vector(t(W) %*% w);
       Ww[r] <- 0;
       Vw <- t(V) %*% w;
       Rw <- pmax(as.vector(Vw - t(H) %*% Ww),0);
       h <- Rw/((t(w) %*% w + epsilon))[1,1];
       H[r,] <- t(h);
     
    }       
}


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

iter <- 0.0;

terminate <- TRUE;

print(normV);

Hnew <- H;
Wnew <- W;

while(terminate) {
    
    if (iter >= maxiter) {
        terminate <- FALSE;
	next;
    }
     
   
    HH <- H %*% t(H); 
    hh <- diag(HH); 
    HH <- HH - diag(hh); 
     
    RH <- (V %*% t(H) - W %*% HH); 
    Wnew[,1] <- pmax(RH[,1]/(hh[1]+epsilon),0); 
    correction <- Wnew[,1]*HH[1,2];
    for (r in 2:K) {	
       RH[,r] <- RH[,r] - correction + (W[,1:(r-1)] %*% as.matrix(HH[1:(r-1),r]));
       Wnew[,r] <- pmax(RH[,r]/(hh[r]+epsilon),0);
       if (r<K) {
           correction  <- Wnew[,1:r] %*% HH[1:r,r+1];
       }
    }
    W <- Wnew;
    
    WW <- t(W) %*% W; 
    ww <- diag(WW);
    WW <- WW - diag(ww);

    VW <- t(V) %*% W;
    RW <- (VW - t(H) %*% WW);
    
    # CONVERGENCE CHECK

    prev_obj<-obj;
    obj <- normV - 2*sum(diag(H %*% VW)) + (sum(diag(HH %*% WW)) + sum(hh*ww)) + epsilon*(sum(hh)+sum(ww)); 
    
    # note we add that last term sum(hh.*ww) to correct for removing diagonals from WW,HH
    
    iter<-iter + 1;  

    print(paste("Iterations<-",iter, "Objective<-", obj));
    check <- (prev_obj < obj);

    if (check) {
                print("obj function increased -- something wrong.\n");
    }
    check <- (abs(prev_obj-obj) < prev_obj*tol) ;

    if (check) {
        terminate <- TRUE;
     }


   # CONVERGENCE CHECK END
    
    Hnew[1,] <- t(pmax(RW[,1]/(ww[1]+epsilon),0));
    correction <- t(Hnew[1,])*WW[1,2];
    for (r in 2:K) {
       if (r>2) {
       RW[,r] <- RW[,r] - correction + (t(H[1:(r-1),]) %*% as.matrix(WW[1:(r-1),r])); 
      }
	else { # due to a weird non-comformable dimensions error - casting between matrices & vectors
		RW[,r] <- RW[,r] - correction + as.vector(H[1:(r-1),] %*% as.matrix(WW[1:(r-1),r]));
	}
       Hnew[r,] <- t(pmax(RW[,r]/(ww[r]+epsilon),0));

       if (r<K) {
          correction <- t(Hnew[1:r,]) %*% WW[1:r,r+1]; 
       }
    }
    H<-Hnew; 
 
}


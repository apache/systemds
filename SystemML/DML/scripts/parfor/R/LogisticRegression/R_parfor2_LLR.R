args <- commandArgs(TRUE)
library("Matrix")
library("doSNOW");

hosts <- c(
	rep("dml1", as.numeric(args[1])),
  rep("dml2", as.numeric(args[1])), 
	rep("dml3", as.numeric(args[1])), 
	rep("dml4", as.numeric(args[1])),
	rep("dml5", as.numeric(args[1]))
	)
cl <- makeCluster(hosts, type = "SOCK")
registerDoSNOW(cl)

# internal parameters
tol = 0.001
eta0 = 0.0001
eta1 = 0.25
eta2 = 0.75
sigma1 = 0.25
sigma2 = 0.5
sigma3 = 4.0
psi = 0.1 

maxiter = 1000 
maxinneriter = 1000

# read training data files
load(file = "/local2/mboehm/parforIn.dat")
N = nrow(X)
D = ncol(X)
numModels = nrow(params);

# initialize variable to store computed models
#wModels = matrix( 0, D, numModels );

wModels = (
foreach( i=1:numModels, .combine=cbind, .multicombine=TRUE ) %dopar%{ #.inorder=FALSE,
  require("Matrix");  
  # retrieve regularizer and weights from parameter settings	
   regul <- as.numeric( params[i,1] );
   wt <- as.numeric( params[i,2] );

   # transpose training data
   tX <- t(X); 
   
   C <- regul * ((y==1) + wt * (y==-1));
   
   #initialize w
   w <- matrix(0,D,1);
   e <- matrix(1,1,1); 

   # matrix multiply training data with weight
   o <- X %*% w
   logistic <- 1.0/(1.0 + exp( -y * o))
   
   # compute objective and gradient
   obj <- 0.5 * t(w) %*% w + sum(-C*log(logistic))
   grad <- w + tX %*% (C*(logistic - 1)*y)
   logisticD <- logistic*(1-logistic)
   delta <- sqrt(sum(grad*grad))
   
   # number of iterations
   iter <- 0
   
   # starting point for CG
   zeros_D <- matrix(0,D,1);
   # VS: change
   zeros_N <- matrix(0,N,1);
   
   # boolean for convergence check
   
   converge <- (delta < tol) | (iter > maxiter)
   norm_r2 <- sum(grad*grad)
   
   # VS: change
   norm_grad <- sqrt(norm_r2)
   norm_grad_initial <- norm_grad
   
   alpha <- t(w) %*% w
   alpha2 <- alpha
   
   #while outer loop not converged
   while(!converge) 
   {
      norm_grad <- sqrt(sum(grad*grad))
      
      #print("-- Outer Iteration = " + iter)
      objScalar = as.numeric(obj)
      #print("     Iterations = " + iter + ", Objective = " + objScalar + ", Gradient Norm = " + norm_grad)
      
      # SOLVE TRUST REGION SUB-PROBLEM
      s <- zeros_D
      os <- zeros_N
      r <- -grad
      d <- r
      inneriter <- 0
      innerconverge <- ( sqrt(sum(r*r)) <= psi * norm_grad) 

      # while inner loop not converged
      while (!innerconverge) 
      {
          inneriter <- inneriter + 1
          norm_r2 <- sum(r*r)
          od <- X %*% d
          Hd <- d + (tX %*% (C*logisticD*od))
          alpha_deno <- t(d) %*% Hd 
          alpha <- norm_r2 / alpha_deno
         
          s <- s + as.numeric(alpha) * d
          os <- os + as.numeric(alpha) * od
        
          sts <- t(s) %*% s
          delta2 <- delta*delta 
          stsScalar <- as.numeric(sts)
          
          shouldBreak <- FALSE;  # to mimic "break" in the following 'if' condition
          if (stsScalar > delta2) 
          {
              #print("      --- cg reaches trust region boundary")
              s <- s - as.numeric(alpha) * d
              os <- os - as.numeric(alpha) * od
              std <- t(s) %*% d
              dtd <- t(d) %*% d
              sts <- t(s) %*% s
              rad <- sqrt(std*std + dtd*(delta2 - sts))
              stdScalar <- as.numeric(std)
              if(stdScalar >= 0) {
               tau <- (delta2 - sts)/(std + rad)
              } 
              else {
               tau <- (rad - std)/dtd
              }
                 
              s <- s + as.numeric(tau) * d
              os <- os + as.numeric(tau) * od
              r <- r - as.numeric(tau) * Hd
              
              #break
              shouldBreak <- TRUE;
              innerconverge <- TRUE;
          } 
       
          if (!shouldBreak) 
          {
              r <- r - as.numeric(alpha) * Hd
              old_norm_r2 <- norm_r2 
              norm_r2 <- sum(r*r)
              beta <- norm_r2/old_norm_r2
              d <- r + beta*d
              innerconverge <- (sqrt(norm_r2) <= psi * norm_grad) | (inneriter > maxinneriter)
          }
        }  
    
        #print("      --- Inner CG Iteration =  " + inneriter)
        # END TRUST REGION SUB-PROBLEM

        # compute rho, update w, obtain delta
        gs <- t(s) %*% grad
        qk <- -0.5*(gs - (t(s) %*% r))
        
        wnew <- w + s 
        onew <- o + os
        logisticnew <- 1.0/(1.0 + exp(-y * onew ))
        objnew <- 0.5 * t(wnew) %*% wnew + sum(-C*log(logisticnew))
        
        actred <- (obj - objnew)
        actredScalar <- as.numeric(actred)
        rho <- actred / qk
        qkScalar <- as.numeric(qk)
        rhoScalar <- as.numeric(rho);
        snorm = sqrt(sum( s * s ))
        #print("     Actual    = " + actredScalar)
        #print("     Predicted = " + qkScalar)
        
        if (iter==0) 
        {
           delta <- min(delta, snorm)
        }

        alpha2 <- objnew - obj - gs
        alpha2Scalar <- as.numeric(alpha2)

        if (alpha2Scalar <= 0) 
        {
           alpha <- sigma3*e
        } else {
           ascalar <- max(sigma1, -0.5*as.numeric(gs)/alpha2Scalar)  
           alpha <- ascalar*e
        }
       
        if (rhoScalar > eta0) 
        {
           w <- wnew
           o <- onew
           grad <- w + tX %*% (C*(logisticnew - 1) * y )
           norm_grad <- sqrt(sum(grad*grad))
           logisticD <- logisticnew * (1 - logisticnew)
           obj <- objnew 
        } 
       
        alphaScalar <- as.numeric(alpha)

        if (rhoScalar < eta0)
        {
           delta <- min(max( alphaScalar , sigma1) * snorm, sigma2 * delta )
        } else {
           if (rhoScalar < eta1)
           {
              delta <- max(sigma1 * delta, min( alphaScalar  * snorm, sigma2 * delta))
           } else { 
              if (rhoScalar < eta2) 
              {
                 delta <- max(sigma1 * delta, min( alphaScalar * snorm, sigma3 * delta))
              } else {
                 delta <- max(delta, min( alphaScalar * snorm, sigma3 * delta))
              }
           }
        } 
        
        o2 <- y * o
        correct <- sum(o2 > 0)        
        accuracy <- correct*100.0/N 
        iter <- iter + 1

        converge <- (norm_grad < tol) | (iter > maxiter)
       
        #print("     Delta =  " + delta)
        #print("     Training Accuracy =  " +  accuracy)
        #print("     Correct =  " + correct)
        #print("     OuterIter =  " + iter)
        #print("     Converge =  " + converge)

   } # OuterWhile

   return (as.matrix(w));
}
)

save(wModels, file = "/local2/mboehm/parforOut.dat");
#writeMM(as(wModels, "CsparseMatrix"), "./tmpout/W");

stopCluster(cl)




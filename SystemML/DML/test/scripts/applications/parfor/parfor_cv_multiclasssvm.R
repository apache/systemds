#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")
library("matrixStats") 



################################################################################

printFoldStatistics = function(stats)
{
   mean_correct_pct = mean( stats[,1])

   #print (" Mean Correct Percentage of the " + nrow( stats) + " Folds: " + mean_correct_pct);
}

################################################################################

scoreMultiClassSVM = function( X, y, W, intercept) 
{
   Nt = nrow(X);
   num_classes = ncol(W)
   b = matrix( 0, 1, num_classes)
   n = ncol(X);

   if (intercept == 1) 
   {
      b = W[n+1,]
   }
       
   ones = matrix( 1, Nt, 1 )
   scores = X %*% W[1:n,] + ones %*% b;                          
                                 
   #predicted_y = which(scores == rowMaxs(scores));
   predicted_y = matrix(0,nrow(scores),1);
   for( i in 1:nrow(scores) )
   {      
      predicted_y[i,1]<-which.max(scores[i,]); 
   }
   
   correct_percentage = sum( ((predicted_y - y)==0.0)) / Nt * 100;
   out_correct_pct = correct_percentage;

   return (out_correct_pct);
}


################################################################################

multiClassSVM = function (X, Y, intercept, num_classes, epsilon, lambda, max_iterations) 
{
   check_X <- sum(X)
   if(check_X == 0){

     print("X has no non-zeros")

   } else {

      num_samples <- nrow(X)
      num_features <- ncol(X)
      
      if (intercept == 1) {
        ones <- matrix( 1, num_samples, 1);
        X <- cbind( X, ones);
      }
      
      iter_class = 1
      
      Y_local <- 2 * ( Y == iter_class ) - 1
      w_class <- matrix( 0, num_features, 1 )
   
      if (intercept == 1) {
         zero_matrix <- matrix( 0, 1, 1 );
         w_class <- t( cbind( t( w_class), zero_matrix));
      }
      
      g_old <- t(X) %*% Y_local
      s <- g_old
      iter <- 0
      continue <- 1
   
      while(continue == 1) {
        # minimizing primal obj along direction s
        step_sz <- 0
        Xd <- X %*% s
        wd <- lambda * sum(w_class * s)
        dd <- lambda * sum(s * s)
        continue1 <- 1
        
        while(continue1 == 1){
         tmp_w <- w_class + step_sz*s
         out <- 1 - Y_local * (X %*% tmp_w)
         sv <- (out > 0)
         out <- out * sv
         g <- wd + step_sz*dd - sum(out * Y_local * Xd)
         h <- dd + sum(Xd * sv * Xd)
         step_sz <- step_sz - g/h
         if (g*g/h < 0.0000000001){
          continue1 = 0
         }
         
        }
       
        #update weights
        w_class <- w_class + step_sz*s
       
        out <- 1 - Y_local * (X %*% w_class)
        sv <- (out > 0)
        out <- sv * out
        obj <- 0.5 * sum(out * out) + lambda/2 * sum(w_class * w_class)
        g_new <- t(X) %*% (out * Y_local) - lambda * w_class
      
        tmp <- sum(s * g_old)
        
        train_acc <- sum( ((Y_local*(X%*%w_class)) >= 0))/num_samples*100
        #print("For class " + iter_class + " iteration " + iter + " training accuracy: " + train_acc)
         
        if((step_sz*tmp < epsilon*obj) | (iter >= max_iterations-1)){
         continue = 0
        }
       
        #non-linear CG step
        be <- sum(g_new * g_new)/sum(g_old * g_old)
        s <- be * s + g_new
        g_old <- g_new
      
        iter <- iter + 1
       }
      
      
      w <- w_class
      iter_class <- iter_class + 1
      
      while(iter_class <= num_classes){
       Y_local <- 2 * (Y == iter_class) - 1
       w_class <- matrix(0, ncol(X), 1)
       if (intercept == 1) {
       	zero_matrix <- matrix(0, 1, 1);
       	w_class <- t(cbind(t(w_class), zero_matrix));
       }
       
       g_old <- t(X) %*% Y_local
       s <- g_old
      
       iter <- 0
       continue <- 1
       while(continue == 1)  {
        # minimizing primal obj along direction s
        step_sz = 0
        Xd <- X %*% s
        wd <- lambda * sum(w_class * s)
        dd <- lambda * sum(s * s)
        continue1 = 1
        while(continue1 == 1){
         tmp_w <- w_class + step_sz*s
         out <- 1 - Y_local * (X %*% tmp_w)
         sv <- (out > 0)
         out <- out * sv
         g <- wd + step_sz*dd - sum(out * Y_local * Xd)
         h <- dd + sum(Xd * sv * Xd)
         step_sz <- step_sz - g/h
         if (g*g/h < 0.0000000001){
          continue1 <- 0
         }
        }
       
        #update weights
        w_class <- w_class + step_sz*s
       
        out <- 1 - Y_local * (X %*% w_class)
        sv <- (out > 0)
        out <- sv * out
        obj <- 0.5 * sum(out * out) + lambda/2 * sum(w_class * w_class)
        g_new <- t(X) %*% (out * Y_local) - lambda * w_class
      
        tmp <- sum(s * g_old)
        
        train_acc <- sum( ((Y_local*(X%*%w_class)) >= 0) )/num_samples*100
        #print("For class " + iter_class + " iteration " + iter + " training accuracy: " + train_acc)
         
        if( ((step_sz*tmp < epsilon*obj) | (iter >= max_iterations-1)) ){
         continue <- 0
        }
       
        #non-linear CG step
        be <- sum(g_new * g_new)/sum(g_old * g_old)
        s <- be * s + g_new
        g_old <- g_new
      
        iter <- iter + 1
       }
      
       w <- cbind(w, w_class) 
       iter_class <- iter_class + 1 
      }
      ret_W <- w
   }
   
   return (ret_W);
}




X <- as.matrix(readMM(paste(args[1], "X.mtx", sep="")))
y <- as.matrix(readMM(paste(args[1], "y.mtx", sep="")))

m = nrow(X);
n = ncol(X);

k = as.numeric(args[2]);

#parameters for model training
intercept = as.numeric(args[3]);
num_classes = as.numeric(args[4]);
epsilon = as.numeric(args[5]);
lambda = as.numeric(args[6]); 
maxiter = as.numeric(args[7]);

#CV
P = as.matrix(readMM(paste(args[1], "P.mtx", sep="")))

ones = matrix(1, 1, n);
stats = matrix(0, k, 1); #k-folds x 1-stats
   
for( i in 1:k )
{
   #prepare train/test fold projections
   vPxi <- (P == i);   #  Select 1/k fraction of the rows   
   mPxi <- (vPxi %*% ones);       #  for the i-th fold TEST set
   #nvPxi <- (P != i);
   #nmPxi <- (nvPxi %*% ones);  #note: inefficient for sparse data  

   #create train/test folds
   Xi <- X * mPxi;  #  Create the TEST set with 1/k of all the rows
   yi <- y * vPxi;  #  Create the labels for the TEST set
   
   nXi <- X - Xi;   #  Create the TRAINING set with (k-1)/k of the rows
   nyi <- y - yi;   #  Create the labels for the TRAINING set
   Xyi <- cbind(Xi,yi); #keep alignment on removeEmpty
   #Xyi <- removeEmpty( target=Xyi, margin="rows" );
   Xyi <- Xyi[rowSums((Xyi==0) | is.na(Xyi)) != ncol(Xyi),];
   Xi <- Xyi[ , 1:n];
   yi <- Xyi[ , n+1];   
   
   nXyi = cbind(nXi,nyi); #keep alignment on removeEmpty
   #nXyi = removeEmpty( target=nXyi, margin="rows" );
   nXyi = nXyi[rowSums((nXyi==0) | is.na(nXyi)) != ncol(nXyi),];
   nXi = nXyi[ , 1:n];
   nyi = nXyi[ , n+1];

   #train multiclass SVM model per fold, use the TRAINING set
   wi = multiClassSVM( nXi, nyi, intercept, num_classes, epsilon, lambda, maxiter)

   #score multiclass SVM model per fold, use the TEST set
   out_correct_pct = scoreMultiClassSVM( Xi, yi, wi, intercept);
   
   stats[i,1] <- out_correct_pct;
}

# print output of stats
printFoldStatistics( stats );

writeMM(as(stats, "CsparseMatrix"), paste(args[8], "stats", sep=""));






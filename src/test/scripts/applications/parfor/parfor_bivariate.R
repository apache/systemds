#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")


D1 <- readMM(paste(args[1], "D.mtx", sep=""))
S11 <- readMM(paste(args[1], "S1.mtx", sep=""))
S21 <- readMM(paste(args[1], "S2.mtx", sep=""))
K11 <- readMM(paste(args[1], "K1.mtx", sep=""))
K21 <- readMM(paste(args[1], "K2.mtx", sep=""))
D <- as.matrix(D1);
S1 <- as.matrix(S11);
S2 <- as.matrix(S21);
K1 <- as.matrix(K11);
K2 <- as.matrix(K21);

numPairs <- ncol(S1) * ncol(S2); # number of attribute pairs (|S1|*|S2|)
maxC <- args[2]; # max number of categories in any categorical attribute

s1size <- ncol(S1);
s2size <- ncol(S2);

# R, chisq, cramers, spearman, eta, anovaf
numstats <- 8;
basestats <- array(0,dim=c(numstats,numPairs)); 
cat_counts <- array(0,dim=c(maxC,numPairs)); 
cat_means <- array(0,dim=c(maxC,numPairs));
cat_vars <- array(0,dim=c(maxC,numPairs));


for( i in 1:s1size ) { 
    a1 <- S1[,i];
    k1 <- K1[1,i];
    A1 <- as.matrix(D[,a1]);

    for( j in 1:s2size ) {
        pairID <-(i-1)*s2size+j;
        a2 <- S2[,j];
        k2 <- K2[1,j];
        A2 <- as.matrix(D[,a2]);
    
        if (k1 == k2) {
            if (k1 == 1) {   
                # scale-scale
                print("scale-scale");
                basestats[1,pairID] <- cor(D[,a1], D[,a2]);
                #basestats[1,pairID] <- cor(A1, A2);
                
                print(basestats[1,pairID]);
            } else {
                # nominal-nominal or ordinal-ordinal
                print("categorical-categorical");
                F <- table(A1,A2);
                cst <- chisq.test(F);
                chi_squared <- as.numeric(cst[1]);
                degFreedom <- (nrow(F)-1)*(ncol(F)-1);
                pValue <- as.numeric(cst[3]);
                q <- min(dim(F));
                W <- sum(F);
                cramers_v <- sqrt(chi_squared/(W*(q-1)));

                basestats[2,pairID] <- chi_squared;
                basestats[3,pairID] <- degFreedom;
                basestats[4,pairID] <- pValue;
                basestats[5,pairID] <- cramers_v;

                if ( k1 == 3 ) {
                    # ordinal-ordinal   
                    print("ordinal-ordinal");
                    basestats[6,pairID] <- cor(A1,A2, method="spearman");
                }
            }
        } 
        else {       
            if (k1 == 1 || k2 == 1) {    
                # Scale-nominal/ordinal
                print("scale-categorical");
                if ( k1 == 1 ) {
                    Av <- as.matrix(A2); 
                    Yv <- as.matrix(A1); 
                }
                else {
                    Av <- as.matrix(A1); 
                    Yv <- as.matrix(A2); 
                }
                
                W <- nrow(Av);
                my <- mean(Yv); 
                varY <- var(Yv);
                
                CFreqs <- as.matrix(table(Av)); 
                CMeans <- as.matrix(aggregate(Yv, by=list(Av), "mean")$V1);
                CVars <- as.matrix(aggregate(Yv, by=list(Av), "var")$V1);
                R <- nrow(CFreqs);
              
                Eta <- sqrt(1 - ( sum((CFreqs-1)*CVars) / ((W-1)*varY) ));
                anova_num <- sum( (CFreqs*(CMeans-my)^2) )/(R-1);
                anova_den <- sum( (CFreqs-1)*CVars )/(W-R);
                ANOVAF <- anova_num/anova_den;

                basestats[7,pairID] <- Eta;
                basestats[8,pairID] <- ANOVAF;

                cat_counts[ 1:length(CFreqs),pairID] <- CFreqs;
                cat_means[ 1:length(CMeans),pairID] <- CMeans;
                cat_vars[ 1:length(CVars),pairID] <- CVars;
            }
            else {
                # nominal-ordinal or ordinal-nominal    
                print("nomial-ordinal"); #TODO should not be same code            
                F <- table(A1,A2);
                cst <- chisq.test(F);
                chi_squared <- as.numeric(cst[1]);
                degFreedom <- (nrow(F)-1)*(ncol(F)-1);
                pValue <- as.numeric(cst[3]);
                q <- min(dim(F));
                W <- sum(F);
                cramers_v <- sqrt(chi_squared/(W*(q-1)));
                
                basestats[2,pairID] <- chi_squared;
                basestats[3,pairID] <- degFreedom;
                basestats[4,pairID] <- pValue;
                basestats[5,pairID] <- cramers_v;
            }
        }
    }
}

writeMM(as(basestats, "CsparseMatrix"), paste(args[3], "bivar.stats", sep=""));
writeMM(as(cat_counts, "CsparseMatrix"), paste(args[3], "category.counts", sep=""));
writeMM(as(cat_means, "CsparseMatrix"), paste(args[3], "category.means", sep=""));
writeMM(as(cat_vars, "CsparseMatrix"), paste(args[3], "category.variances", sep=""));


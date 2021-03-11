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
library("Matrix")

D <- as.matrix(readMM(paste(args[1], "X.mtx", sep="")))
c <- as.matrix(readMM(paste(args[1], "Y.mtx", sep="")))

nClasses <- as.integer(max(c))
varSmoothing <- as.double(args[2])

nSamples <- nrow(D)
nFeatures <- ncol(D)

classInvCovariances <- list()

classMeans <- aggregate(D, by=list(c), FUN= mean)
classMeans <- classMeans[1:nFeatures+1]

classVars <- aggregate(D, by=list(c), FUN=var)
classVars[is.na(classVars)] <- 0
smoothedVar <- varSmoothing * max(classVars) * diag(nFeatures)

classCounts <- aggregate(c, by=list(c), FUN=length)
classCounts <- classCounts[2]
classPriors <- classCounts / nSamples

determinants <- matrix(0, nrow=nClasses, ncol=1)

for (i in 1:nClasses)
{
  classMatrix <- subset(D, c==i)
  covMatrix <- cov(x=classMatrix, use="all.obs")
  covMatrix[is.na(covMatrix)] <- 0
  covMatrix <- covMatrix + smoothedVar
  #determinant <- det(covMatrix)
  #determinants[i] <- det(covMatrix)

  ev <- eigen(covMatrix)
  vecs <- ev$vectors
  vals <- ev$values
  lam <- diag(vals^(-1))
  determinants[i] <- prod(vals)

  invCovMatrix <- vecs %*% lam %*% t(vecs)
  invCovMatrix[is.na(invCovMatrix)] <- 0
  classInvCovariances[[i]] <- invCovMatrix
}


#Calc accuracy
results <- matrix(0, nrow=nSamples, ncol=nClasses)
for (class in 1:nClasses)
{
  for (i in 1:nSamples)
  {
    intermediate <- 0
    meanDiff <- (D[i,] - classMeans[class,])
    intermediate <- -1/2 * log((2*pi)^nFeatures * determinants[class,])
    intermediate <- intermediate - 1/2 * (as.matrix(meanDiff) %*% as.matrix(classInvCovariances[[class]]) 
    %*% t(as.matrix(meanDiff)))
    intermediate <- log(classPriors[class,]) + intermediate
    results[i, class] <- intermediate
  }
}

pred <- max.col(results)
acc <- sum(pred == c) / nSamples * 100
print(paste("Training Accuracy (%): ", acc, sep=""))

classPriors <- data.matrix(classPriors)
classMeans <- data.matrix(classMeans)

#Cbind the inverse covariance matrices, to make them comparable in the unit tests
stackedInvCovs <- classInvCovariances[[1]]
for (i in 2:nClasses)
{
  stackedInvCovs <- cbind(stackedInvCovs, classInvCovariances[[i]])
}

writeMM(as(classPriors, "CsparseMatrix"), paste(args[3], "priors", sep=""));
writeMM(as(classMeans, "CsparseMatrix"), paste(args[3], "means", sep=""));
writeMM(as(determinants, "CsparseMatrix"), paste(args[3], "determinants", sep=""));
writeMM(as(stackedInvCovs, "CsparseMatrix"), paste(args[3], "invcovs", sep=""));

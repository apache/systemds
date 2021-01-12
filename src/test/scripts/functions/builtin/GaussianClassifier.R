library("Matrix")


D <- rbind(c(1, 2, 3, 6), c(4, 3, 2, 5), c(1, 1, 2, 4))

c <- matrix(data=c(1, 2, 1), nrow=3, ncol=1)

nSamples <- nrow(D)
nFeatures <- ncol(D)
nClasses <- max(c)
classInvCovariances <- list()
varSmoothing <- 1e-9

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

  determinant <- det(covMatrix)
  determinants[i] <- det(covMatrix)

  invCovMatrix <- solve(covMatrix)
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
    intermediate <- intermediate - 1/2 * (as.matrix(meanDiff) %*% as.matrix(classInvCovariances[[class]]) %*% t(as.matrix(meanDiff)))
    intermediate <- log(classPriors[class,]) + intermediate
    results[i, class] <- intermediate
  }
}

pred <- max.col(results)
acc <- sum(pred == c) / nSamples * 100
print(paste("Training Accuracy (%): ", acc, sep=""))






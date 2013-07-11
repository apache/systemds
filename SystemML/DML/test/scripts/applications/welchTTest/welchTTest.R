args <- commandArgs(TRUE)
library(Matrix)

posSamples = readMM(paste(args[1], "posSamples.mtx", sep=""))
negSamples = readMM(paste(args[1], "negSamples.mtx", sep=""))

#computing sample sizes
posSampleSize = nrow(posSamples)
negSampleSize = nrow(negSamples)

#computing means
posSampleMeans = colMeans(posSamples)
negSampleMeans = colMeans(negSamples)

#computing (unbiased) variances
posSampleVariances = (colSums(posSamples^2) - posSampleSize * posSampleMeans^2) / (posSampleSize-1)
negSampleVariances = (colSums(negSamples^2) - negSampleSize * negSampleMeans^2) / (negSampleSize-1)

#computing t-statistics and degrees of freedom
t_statistics = (posSampleMeans - negSampleMeans) / sqrt(posSampleVariances/posSampleSize + negSampleVariances/negSampleSize)
degrees_of_freedom = round(((posSampleVariances/posSampleSize + negSampleVariances/negSampleSize) ^ 2) / (posSampleVariances^2/(posSampleSize^2 * (posSampleSize-1)) + negSampleVariances^2/(negSampleSize^2 * (negSampleSize-1))))

#R will write a vector as a 1-column matrix, forcing it to write a 1-row matrix
t_statistics_mat = matrix(t_statistics, 1, length(t_statistics))
degrees_of_freedom_mat = matrix(degrees_of_freedom, 1, length(degrees_of_freedom))

writeMM(as(t_statistics_mat, "CsparseMatrix"), paste(args[2], "t_statistics", sep=""))
writeMM(as(degrees_of_freedom_mat, "CsparseMatrix"), paste(args[2], "degrees_of_freedom", sep=""))

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
args<-commandArgs(TRUE)
options(digits=22)
library("Matrix")
library(mclust, quietly = TRUE)
library("matrixStats") 

# X = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
# X = iris[,1:4]
X = cbind(iris[,1:4],iris[,1:4])
fit =  Mclust(X, modelType = args[2], G=args[3])
summary(fit)
C = fit$z
fit$z
print("this is max")
o = rowMaxs(fit$z)
o
out = o < 0.7
out = as.double(out)
out
fit$df
writeMM(as(C, "CsparseMatrix"), paste(args[4], "B", sep=""))
writeMM(as(out, "CsparseMatrix"), paste(args[4], "O", sep=""))
# library(mclust)

# g <- 3
# dat <- X
# p <- ncol(dat)
# n <- nrow(dat)
# k_fit <- kmeans(dat, centers=g)

# par <- vector("list", g)
# par$pro <- k_fit$size/n
# par$mean <- t(k_fit$centers)

# sigma <- array(NA, c(p, p, g))
# new <- as.data.frame(cbind(dat, k_fit$cluster))
# for (i in 1 : g) {
  
  # subdata <- subset(new[, 1 : p], new[, (p+1)]==i) 
  # sigma[,, i] <- cov(subdata)
# }

# variance <- mclustVariance("EEE", d = p, G = g)
# par$variance <- variance
# par$variance$sigma <- sigma

# kk <- em(modelName = "EEE", data = dat, parameters = par)
# C = kk$z

# C
# o = rowMaxs(kk$z)
# o
# out = o < 0.7
# # out = as.double(out)
# out = rowSums(kk$parameters$mean)
# out
 # writeMM(as(C, "CsparseMatrix"), paste(args[4], "B", sep=""))
 # writeMM(as(out, "CsparseMatrix"), paste(args[4], "O", sep=""))
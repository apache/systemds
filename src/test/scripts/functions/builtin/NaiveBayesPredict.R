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

D = as.matrix(readMM(paste(args[1], "D.mtx", sep="")))
C = as.matrix(readMM(paste(args[1], "C.mtx", sep="")))

# reading input args
numClasses = as.integer(args[2])
laplace = as.double(args[3])

# divide D into "train" and "test" data
numRows = nrow(D)
trainSize = numRows * 0.8
trainData = D[1:trainSize,]
testData = D[(trainSize+1):numRows,]

C = C[1:trainSize,]
numFeatures = ncol(trainData)

# Compute conditionals
# Compute the feature counts for each class
classFeatureCounts = matrix(0, numClasses, numFeatures)
for (i in 1:numFeatures) {
  Col = trainData[,i]
  classFeatureCounts[,i] = aggregate(as.vector(Col), by=list(as.vector(C)), FUN=sum)[,2]
}

# Compute the total feature count for each class
# and add the number of features to this sum
# for subsequent regularization (Laplace's rule)
classSums = rowSums(classFeatureCounts) + numFeatures * laplace

# Compute class conditional probabilities
ones = matrix(1, 1, numFeatures)
repClassSums = classSums %*% ones
class_conditionals = (classFeatureCounts + laplace) / repClassSums

# Compute class priors
class_counts = aggregate(as.vector(C), by=list(as.vector(C)), FUN=length)[,2]
class_prior = class_counts / trainSize;

# Compute accuracy on training set
ones = matrix(1, nrow(testData), 1)
testData_w_ones = cbind(testData, ones)
model = cbind(class_conditionals, class_prior)
YRaw = testData_w_ones %*% t(log(model))

Y = max.col(YRaw, ties.method="last");

# write out the predict
writeMM(as(YRaw, "CsparseMatrix"), paste(args[4], "YRaw", sep=""))
writeMM(as(Y, "CsparseMatrix"), paste(args[4], "Y", sep=""))


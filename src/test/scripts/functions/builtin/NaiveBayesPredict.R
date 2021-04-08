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
probabilities = as.double(args[3])

numRows = nrow(D)
numFeatures = ncol(D)

# Compute conditionals
# Compute the feature counts for each class
classFeatureCounts = matrix(0, numClasses, numFeatures)
for (i in 1:numFeatures) {
  Col = D[,i]
  classFeatureCounts[,i] = aggregate(as.vector(Col), by=list(as.vector(C)), FUN=sum)[,2]
}

# Compute the total feature count for each class
# and add the number of features to this sum
# for subsequent regularization (Laplace's rule)
classSums = rowSums(classFeatureCounts) + numFeatures*probabilities

# Compute class conditional probabilities
ones = matrix(1, 1, numFeatures)
repClassSums = classSums %*% ones
class_conditionals = (classFeatureCounts + probabilities) / repClassSums

# Compute class priors
class_counts = aggregate(as.vector(C), by=list(as.vector(C)), FUN=length)[,2]
class_prior = class_counts / numRows;

# Compute accuracy on training set
ones = matrix(1, numRows, 1)
D_w_ones = cbind(D, ones)
model = cbind(class_conditionals, class_prior)
YRaw = D_w_ones %*% t(log(model))

if(min(C) < 1)
  stop("Stopping due to invalid argument: Label vector (C) must be recoded")

Y = max.col(YRaw,ties.method="last");
acc = sum(Y == C) / numRows * 100;
print(paste("Training Accuracy (%): ", acc, sep=""))

# write out the predict
writeMM(as(YRaw, "CsparseMatrix"), paste(args[4], "YRaw", sep=""))
writeMM(as(Y, "CsparseMatrix"), paste(args[4], "Y", sep=""))


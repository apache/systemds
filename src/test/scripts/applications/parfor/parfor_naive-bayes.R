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

D = as.matrix(readMM(paste(args[1], "D.mtx", sep="")))
C = as.matrix(readMM(paste(args[1], "C.mtx", sep="")))

# reading input args
numClasses = as.integer(args[2]);
laplace_correction = 1

numRows = nrow(D)
numFeatures = ncol(D)

# Compute conditionals

# Compute the feature counts for each class
classFeatureCounts = matrix(0, numClasses, numFeatures)
for (i in 1:numFeatures) {
  Col = D[,i]
  classFeatureCounts[,i] = aggregate(as.vector(Col), by=list(as.vector(C)), FUN=sum)[,2];
}

# Compute the total feature count for each class 
# and add the number of features to this sum
# for subsequent regularization (Laplace's rule)
classSums = rowSums(classFeatureCounts) + numFeatures*laplace_correction

# Compute class conditional probabilities
repClassSums = classSums %*% matrix(1,1,numFeatures);
class_conditionals = (classFeatureCounts + laplace_correction) / repClassSums;

# Compute class priors
class_counts = aggregate(as.vector(C), by=list(as.vector(C)), FUN=length)[,2]
class_prior = class_counts / numRows;

# write out the model
writeMM(as(class_prior, "CsparseMatrix"), paste(args[3], "class_prior", sep=""));
writeMM(as(class_conditionals, "CsparseMatrix"), paste(args[3], "class_conditionals", sep=""));

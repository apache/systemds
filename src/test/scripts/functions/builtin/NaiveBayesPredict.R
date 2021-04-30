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
library(naivebayes)

D = as.matrix(readMM(paste(args[1], "D.mtx", sep="")))
C = as.matrix(readMM(paste(args[1], "C.mtx", sep="")))

# divide D into "train" and "test" data
numRows = nrow(D)

trainSize = numRows * 0.8

trainData = D[1:trainSize,]
testData = D[(trainSize+1):numRows,]

colnames(trainData) <- paste0("V", seq_len(ncol(trainData)))
colnames(testData) <- paste0("V", seq_len(ncol(testData)))
y<-factor(C[1:trainSize])

model<- multinomial_naive_bayes(x = trainData, y = y, laplace = 1)
YRaw <- predict(model, newdata = testData, type = "prob")
Y <- max.col(YRaw, ties.method="last")

# write out the predict
writeMM(as(YRaw, "CsparseMatrix"), paste(args[4], "YRaw", sep=""))
writeMM(as(Y, "CsparseMatrix"), paste(args[4], "Y", sep=""))


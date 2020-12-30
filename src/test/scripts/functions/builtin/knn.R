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
# TODO arguments and order
args <- commandArgs(TRUE)
library("Matrix")

# read test data
data_train            <- as.matrix(readMM(paste(args[1], "/X.mtx", sep="")))
data_test             <- as.matrix(readMM(paste(args[1], "/T.mtx", sep="")))
CL                    <- as.matrix(readMM(paste(args[1], "/CL.mtx", sep="")))

# str(data_train)
# str(data_test)
# str(CL)
## depends how i get the labels/catagories, maybe they are in the training/testing set
# data_train_labels     <- as.matrix(read.csv(args[3], stringsAsFactors = FALSE))
# data_test_labels      <- as.matrix(read.csv(args[4], stringsAsFactors = FALSE))

is_continuous <- as.integer(args[2])
K <- as.integer(args[3])

# ---- normalize -----
normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }

if(is_continuous == 1)
{
  # normalize all but last col (last is our target col)
  data_train_n <- as.data.frame(lapply(data_train[1:NCOL(data_train)], normalize))
  data_test_n  <- as.data.frame(lapply(data_test[1:NCOL(data_test)], normalize))
}

# get the labels, last col
data_train_labels <- CL
# data_train_labels     <- CL[1:nrow(data_train),1] #data_train[1:NROW(data_train), NCOL(data_train)]
# data_test_labels      <- CL[nrow(data_train)+1:nrow(data_test),1]

table(CL)
#print(K)
#print(dim(data_train))
#print(dim(data_test))
#print(dim(CL))

# ------ training -------
#install.packages("class")
library(class)
test_pred <- knn(train= data_train, test= data_test, cl= data_train_labels, k=K)
print("-----------")
print(test_pred)
print("-----------")
# NNR is native NNR, do we realy need to test that?
writeMM(as(R, "CsparseMatrix"), paste(args[1], "PR", sep=""));

## feature importance with random forest
#install.packages("randomForest")
#library(randomForest)
#rf <- randomForest(as.factor(CL)~., data=dat)
#rf$importance

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


# ------ training -------
install.packages("class")
library(class)
test_pred <- knn(train=data_train, test=data_test, cl=CL, k=K)
print("-----------")
print(test_pred)
print("-----------")
writeMM(as(test_pred, "CsparseMatrix"), paste(args[4], "B", sep=""));

## feature importance with random forest
#install.packages("randomForest")
#library(randomForest)
#rf <- randomForest(as.factor(CL)~., data=dat)
#rf$importance

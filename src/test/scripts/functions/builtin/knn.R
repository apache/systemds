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

# read test data
data_train            <- as.matrix(readMM(paste(args[1], "X.mtx", sep="")))
data_test             <- as.matrix(readMM(paste(args[1], "T.mtx", sep="")))

## depends how i get the labels/catagories, maybe they are in the training/testing set
# data_train_labels     <- as.matrix(read.csv(args[3], stringsAsFactors = FALSE))
# data_test_labels      <- as.matrix(read.csv(args[4], stringsAsFactors = FALSE))

continuous <- as.integer(args[2])
K <- as.integer(args[3])

# ---- normalize -----
normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }

if(continuous == 1)
{
  data_train_n <- as.data.frame(lapply(data_train, normalize))
  data_test_n  <- as.data.frame(lapply(data_test, normalize))
}

# ------ training -------
#-- install.packages("class")
#-- library(class)
#Find the number of observation
#-- NROW(data_train_labels)
# k is the square root of number of observation, is that correct?
#-- test_pred <- knn(train = data_train, test =data_test,cl = data_train_labels, k=floor(sqrt(NROW(data_train_labels))))
#--
#-- #Calculate the proportion of correct classification for k = 26, 27
#-- accuracy <- 100 * sum(data_test_labels == pred)/NROW(data_test_labels)
#--
#-- accuracy
#--
#-- # Check prediction against actual value in tabular form
#-- table(test_pred ,data_test_labels)
#-- writeMM(as(test_pred, "CsparseMatrix"), paste(args[4], "B", sep=""))

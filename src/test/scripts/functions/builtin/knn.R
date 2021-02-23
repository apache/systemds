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

is_continuous <- as.integer(args[2])
K <- as.integer(args[3])

library(FNN);
set.seed(10);
tmp_data = rbind(data_train, data_test);
knn_neighbors <- get.knn(tmp_data, k=K);
knn_neighbors <- (tail(knn_neighbors$nn.index, NROW(data_test)));
writeMM(as(knn_neighbors, "CsparseMatrix"), paste(args[4], "NNR", sep=""));


# ------ training -------
library(class)

set.seed(10);
test_pred <- knn(train=data_train, test=data_test, cl=CL, k=K);
print("test_pred:")
print(test_pred)
PR_val <- matrix( , nrow=0, ncol=NCOL(data_test));
for(i in 1:NROW(data_test)) {
  PR_val <- rbind(PR_val, data_train[test_pred[i] , ])
}
writeMM(as(PR_val, "CsparseMatrix"), paste(args[4], "PR", sep=""));

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
library("nnet")

X = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
Y = as.matrix(readMM(paste(args[1], "B.mtx", sep="")))
X_test = as.matrix(readMM(paste(args[1], "C.mtx", sep="")))
Y_test = as.matrix(readMM(paste(args[1], "D.mtx", sep="")))

X = cbind(Y, X)
X_test = cbind(Y_test, X_test)
X = as.data.frame(X)
# set a baseline variable
X$V1 <- relevel(as.factor(X$V1), ref = "3")
X_test = as.data.frame(X_test)
model = multinom(V1~., data = X) # train model
pred <- predict(model, newdata = X_test, "class") # predict unknown data
acc = (sum(pred == Y_test)/nrow(Y_test))*100

writeMM(as(as.matrix(acc), "CsparseMatrix"), paste(args[2], "O", sep=""))
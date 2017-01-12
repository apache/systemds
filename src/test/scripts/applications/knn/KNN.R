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

#parse the args
# JUnit test class: dml.test.integration.applications.KNNTest.java
# Intended to solve knn using R, in order to compare against the DML implementation
# INPUT 1: Matrix X [rows, columns]
# INPUT 2: Matrix y [rows, 1]

args <- commandArgs (TRUE);

library ("Matrix")
library(Matrix)
library(class)

options (warn = -1);

train <- readMM (args[1]);  # (paste (args[1], "X.mtx", sep=""));
test <- readMM (args[1]);  # (paste (args[1], "y.mtx", sep=""));
train
#test
target <- readMM (args[2])
rowNumbs <-  args[4]
cl<-as.factor(c(target[1:rowNumbs]))
print("R input target values:")
cl

selfknn <- function (train, test, cl, k = 1) {
    apply(test,1,function(x){ onerecordknn(train,x,cl,k)})
}
find.lastmax <-
onerecordknn <- function (train, record, cl, k = 1){
    distance = apply(train,1,function(x){sqrt(sum((x-record)^2))})
    #distance = apply(train,1,function(x){sum(abs(x-record))})
    kvalue = cl[order(distance)[1:k]]
    #print(kvalue)
    print(order(distance)[1:k])
    print(sort(distance)[1:k])
    table1 <- table(cl[order(distance)[1:k]])

    maxResult=0
    for(i in length(table1):1){
          # print("item")
           print(table1[i])
           if (table1[i]> maxResult)
                maxResult=table1[i];

    }
    # res<-names(which.max(table1))
    res<-names(maxResult)
    res
}

pr.test <- factor(selfknn(train,test,cl,5))


# pr.test <- knn(train, test, cl, k=5, prob=FALSE)
# print("R predict values:")
# pr.test

dat <- as.numeric(data.matrix(pr.test))
dat
writeMM(as (dat, "CsparseMatrix"), args[3], format = "text");

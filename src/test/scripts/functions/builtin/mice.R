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
library(mice)
library("Matrix")
library(dplyr)

d <- read.csv(args[1], header=FALSE )
mass <- as.matrix(readMM(paste(args[2], "M.mtx", sep="")));

if(sum(mass) == ncol(d))
{
d = d[,3:4]
mass = mass[1,3:4]
meth=""
  for(i in 1: 2) {
      d[[names(d)[i]]] =  as.factor(d[[names(d)[i]]]); 
      meth = c(meth, "polyreg")
    }
  
  meth=meth[-1]

  #impute
  imputeD <- mice(d,where = is.na(d), method = meth, m=3)
  R = data.frame(complete(imputeD,3))
  c = select_if(R, is.factor)

  # convert factor into numeric before casting to matrix
  c =  sapply(c, function(x) as.numeric(as.character(x)))
  writeMM(as(as.matrix(c), "CsparseMatrix"), paste(args[3], "C", sep=""));
} else if (sum(mass) == 0)
{
  print("Generating R witout cat")
  imputeD <- mice(d,where = is.na(d), method = "norm.predict", m=3)
  R = data.frame(complete(imputeD,3))
  n = select_if(R, is.numeric)
  writeMM(as(as.matrix(n), "CsparseMatrix"), paste(args[3], "N", sep=""));  
} else {
  meth=""
  for(i in 1: ncol(mass)) {
    if(as.integer(mass[1,i]) == 1)  {
      d[[names(d)[i]]] =  as.factor(d[[names(d)[i]]]); 
      meth = c(meth, "polyreg")
    } else meth = c(meth, "norm.predict")
  }

  meth=meth[-1]
  # set the prediction matrix
  pred <- make.predictorMatrix(d)
  pred = pred * diag(1, ncol(mass))

  pred[names(d)[1], names(d)[2]] = 1
  pred[names(d)[2], names(d)[1]] = 1

  pred[names(d)[1], names(d)[4]] = 1
  pred[names(d)[4], names(d)[1]] = 1

  pred[names(d)[2], names(d)[4]] = 1
  pred[names(d)[4], names(d)[2]] = 1

  pred[names(d)[3], names(d)[4]] = 1
  pred[names(d)[4], names(d)[3]] = 1


#impute
  imputeD <- mice(d,where = is.na(d), method = meth, m=3,  pred = pred)
  R = data.frame(complete(imputeD,3))
  c = select_if(R, is.factor)
  # convert factor into numeric before casting to matrix
  c =  sapply(c, function(x) as.numeric(as.character(x)))
  n = select_if(R, is.numeric)
  writeMM(as(as.matrix(c), "CsparseMatrix"), paste(args[3], "C", sep=""));
  writeMM(as(as.matrix(n), "CsparseMatrix"), paste(args[3], "N", sep=""));
}

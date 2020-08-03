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
library(mclust, quietly = TRUE)
library("matrixStats") 

X = iris[,1:4]
fit =  Mclust(X, modelType = args[3], G=args[2])
prob = fit$z
out = rowMaxs(fit$z) < 0.7
out = as.double(out)

writeMM(as(prob, "CsparseMatrix"), paste(args[4], "B", sep=""))
writeMM(as(out, "CsparseMatrix"), paste(args[4], "O", sep=""))
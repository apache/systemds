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
library("Matrix")
library("dbscan")

X = as.matrix(readMM(paste(args[1], "A.mtx", sep="")));
Y = as.matrix(readMM(paste(args[2], "B.mtx", sep="")));
eps = as.double(args[3]);
minPts = as.integer(args[4]);
dbModel = dbscan(X, eps, minPts);

cleanMatr = matrix(, nrow = nrow(X), ncol = 3)
for(i in 1:nrow(X)) {
  if(dbModel$cluster[i] > 0) {
    cleanMatr[i,] = X[i,]
  }
}

cleanMatr = cleanMatr[rowSums(is.na(cleanMatr)) != ncol(cleanMatr),]

dbModelClean = dbscan(cleanMatr, eps, minPts);

Z = predict(dbModelClean, Y, data = cleanMatr);
Z[Z > 0] = 1;
writeMM(as(Z, "CsparseMatrix"), paste(args[5], "C", sep=""));

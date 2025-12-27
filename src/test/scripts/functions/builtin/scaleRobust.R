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

library("Matrix")

args <- commandArgs(TRUE)
options(digits=22)


X = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
colnames(X) = colnames(X, do.NULL=FALSE, prefix="C")
Y = X

for (j in 1:ncol(X)) {
  col = X[, j]
  med = quantile(col, probs=0.5, type=1, names=FALSE, na.rm=FALSE)
  q1  = quantile(col, probs=0.25, type=1, names=FALSE, na.rm=FALSE)
  q3  = quantile(col, probs=0.75, type=1, names=FALSE, na.rm=FALSE)
  iqr = q3 - q1
  if (iqr == 0 || is.nan(iqr)) iqr = 1
  Y[, j] = (col - med) / iqr
}

writeMM(as(Y, "CsparseMatrix"), paste(args[2], "B", sep=""))

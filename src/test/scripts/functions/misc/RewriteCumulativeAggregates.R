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
options(digits=22)
library("Matrix")

X = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
num = as.integer(args[2]);

#note: cumsum and rev only over vectors
if( num == 1 ) {
  R = lower.tri(X,diag=TRUE) * X;
} else if( num == 2 ) {
  A = X[seq(nrow(X),1),]
  R = apply(A, 2, cumsum);
  R = R[seq(nrow(X),1),]
} else if( num == 3 ) {
  R = t(as.matrix(colSums(apply(X, 2, cumsum))));
} else if( num == 4 ) {
  if( nrow(X)==1 ) {
    R = X;
  } else {
    R = apply(X, 2, cumsum);
  }
}

writeMM(as(R, "CsparseMatrix"), paste(args[3], "R", sep="")); 

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

X = as.matrix(readMM(paste(args[1], "V.mtx", sep="")))
N = 200;

R = matrix(0, ceiling(nrow(X)/N), 1); 
for( bi in 1:ceiling(nrow(X)/N)) {
   Xbi = X[(7+(bi-1)*N+1):min(bi*N,nrow(X)),];   
   R[bi,1] = sum(Xbi); 
}   

writeMM(as(R, "CsparseMatrix"), paste(args[2], "Rout", sep=""));

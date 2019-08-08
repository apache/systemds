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

U = matrix( 4, 4000, 10)
V = matrix( 5, 10, 2000)
X = U %*% V
X[1:3500,] = matrix(0,3500,2000);

T1 = rowSums(U^2)%*%matrix(1,1,ncol(V));
T2 = matrix(1,nrow(U),1)%*%t(as.matrix(colSums(V^2)))
D = sqrt(-2 * U %*% V + T1 + T2);

s = sum(rowSums((X != 0) * (X - D))^2)
S = as.matrix(s); 

writeMM(as(S, "CsparseMatrix"), paste(args[2], "S", sep="")); 
 
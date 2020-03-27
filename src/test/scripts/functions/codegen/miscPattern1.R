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

X = matrix(1, 1100, 2200);
Y = matrix(2, 1100, 2200);
U = matrix(3, 1100, 10);
V = matrix(4, 2200, 10)
X[4:900,3:1000] = matrix(0, 897, 998);

R1 = Y + (X * U%*%t(V));
R2 = as.matrix(sum(R1));

writeMM(as(R2, "CsparseMatrix"), paste(args[1], "S", sep="")); 

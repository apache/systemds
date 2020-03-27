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

V = matrix(seq(0,14), 5, 3, byrow=TRUE);
X = matrix(seq(1,15), 5, 3, byrow=TRUE);
Y = matrix(seq(2,16), 5, 3, byrow=TRUE);
Z = matrix(seq(3,17), 5, 3, byrow=TRUE);

#disjoint partitions with transitive partial shared reads
r1 = sum(V * X);
r2 = sum(Y * Z);
r3 = sum(X * Y * Z);
S = as.matrix(r1+r2+r3);

writeMM(as(S, "CsparseMatrix"), paste(args[2], "S", sep="")); 

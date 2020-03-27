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

X = matrix(1, 1500, 100) * matrix(1,1500,1)%*%t(seq(1,100));

X0 = X - 0.5;
X1 = X / rowSums(X0)%*%matrix(1,1,100);
X2 = abs(X1 * 0.5);
X3 = X1 / rowSums(X2)%*%matrix(1,1,100);

R = as.matrix(sum(X3));

writeMM(as(R,"CsparseMatrix"), paste(args[2], "S", sep=""));

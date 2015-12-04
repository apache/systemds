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

A1=readMM(paste(args[1], "A.mtx", sep=""))
A = as.matrix(A1);
B1=readMM(paste(args[1], "B1.mtx", sep=""))
B1 = as.matrix(B1);
B2=readMM(paste(args[1], "B2.mtx", sep=""))
B2 = as.matrix(B2);
C=cbind2(A, B1)
C=cbind2(C, B2)
writeMM(as(C,"CsparseMatrix"), paste(args[2], "C", sep=""), format="text")

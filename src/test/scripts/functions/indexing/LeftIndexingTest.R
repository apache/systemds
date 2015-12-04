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
B1=readMM(paste(args[1], "B.mtx", sep=""))
C1=readMM(paste(args[1], "C.mtx", sep=""))
D1=readMM(paste(args[1], "D.mtx", sep=""))
A=as.matrix(A1);
B=as.matrix(B1);
C=as.matrix(C1);
D=as.matrix(D1);

A[args[2]:args[3],args[4]:args[5]]=0
A[args[2]:args[3],args[4]:args[5]]=B
writeMM(as(A,"CsparseMatrix"), paste(args[6], "AB", sep=""), format="text")
A[1:args[3],args[4]:ncol(A)]=0
A[1:args[3],args[4]:ncol(A)]=C
writeMM(as(A,"CsparseMatrix"), paste(args[6], "AC", sep=""), format="text")
A[,args[4]:args[5]]=0
A[,args[4]:args[5]]=D
writeMM(as(A,"CsparseMatrix"), paste(args[6], "AD", sep=""), format="text")
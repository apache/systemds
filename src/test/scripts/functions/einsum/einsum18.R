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
library("matrixStats")
library("einsum")

A0 = matrix(seq(1,250), 10, 25, byrow=TRUE) * 0.0001
A1 = matrix(seq(1,200), 10, 20, byrow=TRUE) * 0.0001
A2 = matrix(seq(1,30), 10, 3, byrow=TRUE) * 0.0001
A3 = matrix(seq(1,500), 25, 20, byrow=TRUE) * 0.0001
A4 = matrix(seq(1,33), 3, 11, byrow=TRUE) * 0.0001
A5 = seq(1,11) * 0.0001
A6 = seq(1,3) * 0.0001

R = einsum("fx,fg,fz,xg,pq,q,p->zp", A0, A1, A2, A3, A4, A5, A6)

writeMM(as(R, "CsparseMatrix"), paste(args[2], "S", sep="")); 

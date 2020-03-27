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

N = as.integer(args[1]);
A = as.matrix(seq(N,1,-1));

op = ifelse(args[2]=="lt","<",ifelse(args[2]=="lte","<=",ifelse(args[2]=="gt",">",ifelse(args[2]=="gte",">=", args[2])))) 

C = as.matrix(outer(A, t(A), op));
C = matrix(C, nrow=nrow(A), ncol=nrow(A), byrow=FALSE)

writeMM(as(C, "CsparseMatrix"), paste(args[3], "C", sep="")); 

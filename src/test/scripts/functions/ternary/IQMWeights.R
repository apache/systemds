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

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))

#without weights (assumes weights of 1)
m = nrow(A);
S = sort(A)
q25d=m*0.25
q75d=m*0.75
q25i=ceiling(q25d)
q75i=ceiling(q75d)
iqm = sum(S[(q25i+1):q75i])
iqm = iqm + (q25i-q25d)*S[q25i] - (q75i-q75d)*S[q75i]
iqm = iqm/(m*0.5)

miqm = as.matrix(iqm);

writeMM(as(miqm, "CsparseMatrix"), paste(args[3], "R", sep="")); 



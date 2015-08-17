#-------------------------------------------------------------
#
# (C) Copyright IBM Corp. 2010, 2015
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#-------------------------------------------------------------


args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")

A1=readMM(paste(args[1], "A.mtx", sep=""))
A = as.matrix(A1);

B=A[args[2]:args[3],args[4]:args[5]]
C=A[1:args[3],args[4]:ncol(A)]
D=A[,args[4]:args[5]]
writeMM(as(B,"CsparseMatrix"), paste(args[6], "B", sep=""), format="text")
writeMM(as(C,"CsparseMatrix"), paste(args[6], "C", sep=""), format="text")
writeMM(as(D,"CsparseMatrix"), paste(args[6], "D", sep=""), format="text")

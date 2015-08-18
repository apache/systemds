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

X = as.matrix(readMM(paste(args[1], "X.mtx", sep="")))
U = as.matrix(readMM(paste(args[1], "U.mtx", sep="")))
V = as.matrix(readMM(paste(args[1], "V.mtx", sep="")))

UV = -(U%*%t(V));
R = X * (1/(1 + exp(-UV)));

writeMM(as(R, "CsparseMatrix"), paste(args[2], "R", sep="")); 



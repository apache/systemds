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

X = matrix(1, 1000, 3);
B = matrix(1, 1000, 2);
C = matrix(7, 1000, 1);
D = matrix(3, 1000, 1);

E = cbind(X [, 1 : 2], B) * ((C * (1 - D))%*%matrix(1,1,4));
X = X * C%*%matrix(1,1,3);
n = nrow (X);

R = X + sum(E) + n;

cat(sum(R))

writeMM(as(R, "CsparseMatrix"), paste(args[1], "R", sep="")); 
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
#X= matrix( seq(1,25), 5, 5, byrow = TRUE)
X = matrix(  c(1,2,3,4,5,6,7,8,9), nrow=3, ncol=3, byrow = TRUE)
w=matrix(  c(1,1,1,2,2,2,3,3,3), nrow=3, ncol=3, byrow = TRUE)
z=matrix(  c(3,3,3,3,3,3,3,3,3), nrow=3, ncol=3, byrow = TRUE)
#S= X*as.matrix(X>0)
#S=7 + (1 / exp(X) )
G = abs(exp(X))
Y=10 + floor(round(abs((X/w)+z)))
S = G + Y
writeMM(as(S, "CsparseMatrix"), paste(args[2], "S", sep="")); 
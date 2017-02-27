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
#X= matrix( 1, 100, 100)
#X = matrix(  c(1,2,3,4,5,6,7,8,9), nrow=3, ncol=3, byrow = TRUE)
#X= matrix(  c(0,0,3,4,0,0,0,8,0),  nrow=3, ncol=3, byrow = TRUE)
#Y= matrix( c(2,2,2,3,3,3,1,1,1), nrow=3, ncol=3, byrow = TRUE)
#X= matrix(1, 1001, 1001)

X= matrix( seq(1,4000000), 2000,2000, byrow=TRUE)
#X= matrix(1, 2000,2000, byrow=TRUE)

Y= matrix( 2, 2000, 2000)
#S= X*(1-X)
lamda = 4000

S=round(abs(X+lamda))+5
#S=sum(X+Y+5)
#S=round(X+(X+9.5))
#print(S)
writeMM(as(S, "CsparseMatrix"), paste(args[2], "S", sep="")); 
 
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

X <- as.matrix(readMM(paste(args[1], "X.mtx", sep="")))

if( as.integer(args[2])==1 )
{
   v = matrix(1,nrow(X),1);
   X = as.matrix(cbind(X, v));
}

if( as.integer(args[2])!=1 )
{
   v = matrix(1,nrow(X),1);
   X = as.matrix(cbind(X, v));
} else
{
   v1 = matrix(1,nrow(X),1);
   X = as.matrix(cbind(X, v1));
   v2 = matrix(1,nrow(X),1);
   X = as.matrix(cbind(X, v2));
}

writeMM(as(X, "CsparseMatrix"), paste(args[3], "X", sep="")); 
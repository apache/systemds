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

A = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
c = as.integer(args[2]);
mStep = as.logical(args[3]);

s = 0;
if( mStep ) {
   for( i in seq(nrow(A),1,-7) ) {
      s = s + A[i,1] + c;
   }
} else {
   for( i in nrow(A):1 ) {
      s = s + A[i,1] + c;
   }
}

R = as.matrix(s);
writeMM(as(R, "CsparseMatrix"), paste(args[4], "R", sep="")); 

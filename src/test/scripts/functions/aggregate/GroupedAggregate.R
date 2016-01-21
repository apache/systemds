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

library("Matrix")
library("moments")

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
B <- as.matrix(readMM(paste(args[1], "B.mtx", sep="")));
fn = as.integer(args[2]);

if( fn==0 )
{
   C = aggregate(as.vector(A), by=list(as.vector(B)), FUN=sum)[,2]
}

if( fn==1 ) 
{
   C = aggregate(as.vector(A), by=list(as.vector(B)), FUN=length)[,2]
}

if( fn==2 ) 
{
   C = aggregate(as.vector(A), by=list(as.vector(B)), FUN=mean)[,2]
}

if( fn==3 ) 
{
   C = aggregate(as.vector(A), by=list(as.vector(B)), FUN=var)[,2]
}

if( fn==4 ) 
{
   C = aggregate(as.vector(A), by=list(as.vector(B)), FUN=moment, order=3, central=TRUE)[,2]
}

if( fn==5 ) 
{
   C = aggregate(as.vector(A), by=list(as.vector(B)), FUN=moment, order=4, central=TRUE)[,2]
}

writeMM(as(C, "CsparseMatrix"), paste(args[3], "C", sep="")); 
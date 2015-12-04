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

X <- readMM(paste(args[1], "X.mtx", sep=""))

type = as.integer(args[2]);

R = X;

if( type==0 ){
  R = as.matrix( sum(X) );
}
if( type==1 ){
  R = round(X);
}
if( type==2 ){
  R = t(X); 
}
if( type==3 ){
  R = X*(X-1);
}
if( type==4 ){
  R = (X-1)*X;
}
if( type==5 ){
  R = X+(X-1);
}
if( type==6 ){
  R = (X-1)+X;
}
if( type==7 ){
  R = X-(X+2);
}
if( type==8 ){
  R = (X+2)-X;
}
if( type==9 ){
  R = X%*%(X-1);
}
if( type==10 ){
  R = (X-1)%*%X;
}
if( type==11 ){
  R = X[1:(nrow(X)-1), 1:(ncol(X)-1)];
}
if( type==12 ){
  X[1,] = X[2,];
  R = X;
}

writeMM(as(R, "CsparseMatrix"), paste(args[3], "R", sep="")); 
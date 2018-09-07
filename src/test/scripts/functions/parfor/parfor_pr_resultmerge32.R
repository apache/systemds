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

V1 <- readMM(paste(args[1], "V.mtx", sep=""))
V <- as.matrix(V1);
m <- nrow(V); 
n <- ncol(V); 

R1 <- matrix(0,m,n);
R2 <- matrix(0,m,n);
R3 <- matrix(0,m,n);
R4 <- matrix(0,m,n);
R5 <- matrix(0,m,n);
R6 <- matrix(0,m,n);
R7 <- matrix(0,m,n);
R8 <- matrix(0,m,n);
R9 <- matrix(0,m,n);
R10 <- matrix(0,m,n);
R11 <- matrix(0,m,n);
R12 <- matrix(0,m,n);
R13 <- matrix(0,m,n);
R14 <- matrix(0,m,n);
R15 <- matrix(0,m,n);
R16 <- matrix(0,m,n);
R17 <- matrix(0,m,n);
R18 <- matrix(0,m,n);
R19 <- matrix(0,m,n);
R20 <- matrix(0,m,n);
R21 <- matrix(0,m,n);
R22 <- matrix(0,m,n);
R23 <- matrix(0,m,n);
R24 <- matrix(0,m,n);
R25 <- matrix(0,m,n);
R26 <- matrix(0,m,n);
R27 <- matrix(0,m,n);
R28 <- matrix(0,m,n);
R29 <- matrix(0,m,n);
R30 <- matrix(0,m,n);
R31 <- matrix(0,m,n);
R32 <- matrix(0,m,n);

for( i in 1:n )
{
   X <- V[,i];
   R1[,i] <- X;
   R2[,i] <- X;
   R3[,i] <- X;
   R4[,i] <- X;
   R5[,i] <- X;
   R6[,i] <- X;
   R7[,i] <- X;
   R8[,i] <- X;
   R9[,i] <- X;
   R10[,i] <- X;
   R11[,i] <- X;
   R12[,i] <- X;
   R13[,i] <- X;
   R14[,i] <- X;
   R15[,i] <- X;
   R16[,i] <- X;
   R17[,i] <- X;
   R18[,i] <- X;
   R19[,i] <- X;
   R20[,i] <- X;
   R21[,i] <- X;
   R22[,i] <- X;
   R23[,i] <- X;
   R24[,i] <- X;
   R25[,i] <- X;
   R26[,i] <- X;
   R27[,i] <- X;
   R28[,i] <- X;
   R29[,i] <- X;
   R30[,i] <- X;
   R31[,i] <- X;
   R32[,i] <- X;
}   

R <- R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10 + R11 + R12 + R13 + R14 + R15 + R16 + R17 + R18 + R19 + R20 + R21 + R22 + R23 + R24 + R25 + R26 + R27 + R28 + R29 + R30 + R31 + R32; 
writeMM(as(R, "CsparseMatrix"), paste(args[2], "Rout", sep=""));

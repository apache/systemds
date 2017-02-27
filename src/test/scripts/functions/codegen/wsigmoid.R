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

X= matrix( c(1,0,1,2,2,0,0,3,3,0,0,4), nrow=4, ncol=3, byrow = TRUE)
U= matrix( c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8), nrow=4, ncol=2, byrow = TRUE)
V= matrix( c(0.9,0.12,0.10,0.13,0.11,0.14), nrow=3, ncol=2, byrow = TRUE)
eps = 0.1
S= X*(1/(1+exp(-(U%*%t(V)))));
writeMM(as(S, "CsparseMatrix"), paste(args[2], "S", sep="")); 
 
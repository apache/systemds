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
library("matrixStats")
A = matrix(seq(1, 20), nrow=5, ncol=4, byrow=TRUE)
Y = matrix(seq(1, 12), nrow=4, ncol=3, byrow=TRUE)
X = t(A)

R <- t(t(Y)%*%X)

writeMM(as(R, "CsparseMatrix"), paste(args[1], "R", sep=""));

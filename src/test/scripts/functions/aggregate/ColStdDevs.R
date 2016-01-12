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
library("matrixStats")

X <- as.matrix(readMM(paste(args[1], "X.mtx", sep="")))
colStdDevs <- t(colSds(X))

# R outputs the standard deviation of a single number as NA, whereas
# a more appropriate value would be 0, so replace all NAs with zeros.
# This is useful in the case of colStdDevs of a row vector.
colStdDevs[is.na(colStdDevs)] <- 0

writeMM(as(colStdDevs, "CsparseMatrix"), paste(args[2], "colStdDevs", sep=""));

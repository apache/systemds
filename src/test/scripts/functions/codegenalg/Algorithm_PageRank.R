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

G = readMM(paste(args[1], "G.mtx", sep=""));
p = as.matrix(readMM(paste(args[1], "p.mtx", sep="")));
e = as.matrix(readMM(paste(args[1], "e.mtx", sep="")));
u = as.matrix(readMM(paste(args[1], "u.mtx", sep="")));
alpha = as.double(args[2]);
max_iteration = as.integer(args[3]);
i = 0;

while( i < max_iteration ) {
  p = alpha * (G %*% p) + (1 - alpha) * (e %*% (u %*% p));
  i = i + 1;
}

writeMM(as(p,"CsparseMatrix"), paste(args[4], "p", sep=""));

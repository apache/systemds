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

# Function definition for raSelection in R
raSelection <- function(X, col, op, val) {
  # Determine the operators
  I = switch(op,
              "==" = X[, col] == val,
              "!=" = X[, col] != val,
              "<"  = X[, col] <  val,
              ">"  = X[, col] >  val,
              "<=" = X[, col] <= val,
              ">=" = X[, col] >= val,
              stop("Invalid operator"))

  # Select rows based on the condition
  Y = X[I, , drop = FALSE]

  return(Y)
}

args<-commandArgs(TRUE)
options(digits=22)
library("Matrix")

X = as.matrix(readMM(paste(args[1],"X.mtx",sep="")));
col = as.integer(args[2])
op = args[3]
val = as.numeric(args[4])

result = raSelection(X,col,op,val);
writeMM(as(result,"CsparseMatrix"),paste(args[5],"result",sep=""));
 
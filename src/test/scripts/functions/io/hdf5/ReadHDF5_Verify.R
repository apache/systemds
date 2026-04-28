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
options(digits=22)

library("rhdf5")

Y = h5read(args[1], args[2], native = TRUE)
dims = dim(Y)

if(length(dims) == 1) {
  # convert to a column matrix
  Y_mat = matrix(Y, ncol = 1)
} else if(length(dims) > 2) {
  # flatten everything beyond the first dimension into columns
  perm = c(1, rev(seq(2, length(dims))))
  Y_mat = matrix(aperm(Y, perm), nrow = dims[1], ncol = prod(dims[-1]))
} else {
  # for 2d , systemds treats it the same
  Y_mat = Y
}

writeMM(as(Y_mat, "CsparseMatrix"), paste(args[3], "Y", sep=""))

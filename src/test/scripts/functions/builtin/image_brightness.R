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

args = commandArgs(TRUE)
options(digits=22)
library("Matrix")

adjustBrightness = function(M, val, chan_max)
{
  out = matrix(0, nrow(M), ncol(M));
  for( i in 1:ncol(M) ) 
  {
    col = as.vector(M[,i])
    #col = rev(col);
    col = pmax(0, pmin(col + val, chan_max))
    out[,i] = col;
  }
  return(out)
}

A = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
#print(paste("input dim:", toString(dim(X)),sep=" "))
#print(X)

#B = pmax(0, pmin(A + 123, 255))
B = adjustBrightness(A, 123, 255)

writeMM(as(B, "CsparseMatrix"), paste(args[2], "B", sep=""));
#print(paste("output Bx dim:", toString(dim(Bx)), sep=" "))
#print(Bx)

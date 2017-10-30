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
imgSize=as.integer(args[1])
numImg=as.integer(args[2])
numChannels=as.integer(args[3])

# Assumption: NCHW image format
x=matrix(seq(1, numImg*numChannels*imgSize*imgSize), numImg, numChannels*imgSize*imgSize, byrow=TRUE)
if(as.logical(args[5])) {
	zero_mask = (x - 1.5*mean(x)) > 0 
	x = x * zero_mask
} else {
	x = x - mean(x)
}

output = rowSums(matrix(colSums(x), numChannels, imgSize*imgSize, byrow=TRUE));

writeMM(as(output,"CsparseMatrix"), paste(args[4], "B", sep=""))
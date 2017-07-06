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
poolSize1=as.integer(args[4])
poolSize2=as.integer(args[5])
stride=as.integer(args[6])
pad=as.integer(args[7])
P=as.integer(args[8])
Q=as.integer(args[9])

# Assumption: NCHW image format
x=matrix(seq(1, numImg*numChannels*imgSize*imgSize), numImg, numChannels*imgSize*imgSize, byrow=TRUE)
dout=matrix(seq(1, numImg*numChannels*P*Q), numImg, numChannels*P*Q, byrow=TRUE)
if(as.logical(args[11])) {
	# zero_mask = (x - mean(x)) > 0 
	# x = x * zero_mask
}
if(as.logical(args[12])) {
	# zero_mask = (dout - mean(dout)) > 0 
	# dout = dout * zero_mask
}
x = x - mean(x)
dout = dout - mean(dout)
max_pool_backward <- function(dout, Hout, Wout, X, C,
                    Hin, Win, Hf, Wf, strideh, stridew)
     {
  N = nrow(X)
  
  # Create gradient volume
  dX = matrix(0, N, C*Hin*Win, byrow=TRUE)
  
  # Gradient of max pooling
  for (n in 1:N) {  # all examples
    img = matrix(X[n,], C, Hin*Win, byrow=TRUE)
    dimg = matrix(0, C, Hin*Win, byrow=TRUE)
    for (c in 1:C) {  # all channels
      img_slice = matrix(img[c,], Hin, Win, byrow=TRUE)
      dimg_slice = matrix(0, Hin, Win, byrow=TRUE)
      for (hout in 1:Hout) {  # all output rows
        hin = (hout-1) * strideh + 1
        for (wout in 1:Wout) {  # all output columns
          win = (wout-1) * stridew + 1
          img_slice_patch = img_slice[hin:(hin+Hf-1), win:(win+Wf-1)]
          max_val = max(img_slice_patch)
          max_val_ind = (img_slice_patch == max_val)  # max value indicator
          # gradient passes through only for the max value in this patch
          dimg_slice_patch = max_val_ind * dout[n, (c-1)*Hout*Wout + (hout-1)*Wout + wout]
          dimg_slice[hin:(hin+Hf-1), win:(win+Wf-1)] =
            dimg_slice[hin:(hin+Hf-1), win:(win+Wf-1)] + dimg_slice_patch
        }
      }
      dimg[c,] = matrix(t(dimg_slice), 1, Hin*Win)
    }
    dX[n,] = matrix(t(dimg), 1, C*Hin*Win)
  }
  
  dX
}

output = max_pool_backward(dout, P, Q, x, numChannels, imgSize, imgSize, poolSize1, poolSize2, stride, stride)
writeMM(as(output,"CsparseMatrix"), paste(args[10], "B", sep=""))


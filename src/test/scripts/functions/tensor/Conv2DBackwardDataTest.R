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
imgSize=as.integer(args[1])
numImg=as.integer(args[2])
numChannels=as.integer(args[3])
numFilters=as.integer(args[4])
filterSize=as.integer(args[5])
stride=as.integer(args[6])
pad=as.integer(args[7])
P=as.integer(args[8])
Q=as.integer(args[9])

# Assumption: NCHW image format
w=matrix(seq(1, numFilters*numChannels*filterSize*filterSize), numFilters, numChannels*filterSize*filterSize, byrow=TRUE)
dout=matrix(seq(1, numImg*numFilters*P*Q), numImg, numFilters*P*Q, byrow=TRUE)


col2im <- function(img_cols, C, Hin, Win, Hf, Wf,
                  strideh, stridew, reduction) {

  Hout = as.integer((Hin - Hf) / strideh + 1)
  Wout = as.integer((Win - Wf) / stridew + 1)

  img = matrix(0, C, Hin*Win, byrow=TRUE)  # zeros
  for (hout in 1:Hout) {  # all output rows
    hin = (hout-1) * strideh + 1
    for (wout in 1:Wout) {  # all output columns
      win = (wout-1) * stridew + 1
      # Extract a local patch of the input image corresponding spatially to the filter sizes.
      img_patch = matrix(img_cols[,(hout-1)*Wout + wout], C, Hf*Wf, byrow=TRUE)  # zeros
      for (c in 1:C) {  # all channels
        img_patch_slice = matrix(img_patch[c,], Hf, Wf, byrow=TRUE)  # reshape
        if (reduction == "add") {
          img_slice = matrix(0, Hin, Win, byrow=TRUE)
          img_slice[hin:(hin+Hf-1), win:(win+Wf-1)] = img_patch_slice
          img[c,] = img[c,] + matrix(t(img_slice), 1, Hin*Win)
        } else {
          img_slice = matrix(img[c,], Hin, Win, byrow=TRUE)
          img_slice[hin:(hin+Hf-1), win:(win+Wf-1)] = img_patch_slice
          img[c,] = matrix(t(img_slice), 1, Hin*Win)
        }
      }
    }
  }
  img
}

unpad_image <- function(img_padded, Hin, Win, padh, padw) {
  C = nrow(img_padded)
  img = matrix(0, C, Hin*Win, byrow=TRUE)
  for (c in 1:C) {
    img_padded_slice = matrix(img_padded[c,], (Hin+2*padh), (Win+2*padw), byrow=TRUE)
    img_slice = img_padded_slice[(padh+1):(padh+Hin), (padw+1):(padw+Win)]
    img[c,] = matrix(t(img_slice), 1, Hin*Win)
  }
  img
}

conv2d_backward_data <- function(dout, Hout, Wout,
                    W, N, C, Hin, Win, Hf, Wf,
                    strideh, stridew, padh, padw) {

  F = nrow(W)
  
  # Create gradient volumes
  dX = matrix(0, N, C*Hin*Win, byrow=TRUE)
  
  # Partial derivatives for convolution - im2col implementation
  for (n in 1:N) {  # all examples
    doutn = matrix(dout[n,], F, Hout*Wout, byrow=TRUE)

    # Compute dX
    dXn_padded_cols = t(W) %*% doutn  # shape (C*Hf*Wf, Hout*Wout)
    dXn_padded = col2im(dXn_padded_cols, C, Hin+2*padh, Win+2*padw, Hf, Wf, strideh, stridew, "add")
    dXn = unpad_image(dXn_padded, Hin, Win, padh, padw)
    dX[n,] = matrix(t(dXn), 1, C*Hin*Win)  # reshape
  }
  
  dX
}

dx = conv2d_backward_data(dout, P, Q, w, numImg, numChannels, imgSize, imgSize, filterSize, filterSize, stride, stride, pad, pad);

writeMM(as(dx,"CsparseMatrix"), paste(args[10], "B", sep=""))
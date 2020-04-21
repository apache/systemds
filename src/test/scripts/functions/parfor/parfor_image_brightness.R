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

adjustBrightness = function(M, val, chan_max) {
  out = pmax(0, pmin(M + val, chan_max))
  out = matrix(out, nrow(M), ncol(M), byrow=FALSE)
  return(out)
}

images = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
width = as.integer(args[3])
height = as.integer(args[4])
brightness_adjustments = as.matrix(readMM(paste(args[1], "brightness_adjustments.mtx", sep="")))
num_augmentations = nrow(brightness_adjustments)
augmented_images = matrix(0, num_augmentations * nrow(images), ncol(images))
max_value=255

for (idx in 0:(nrow(images)-1)) {
  i = idx + 1

  image2d = matrix(images[i,], width, height, byrow=TRUE)

  for(a in 1:num_augmentations) {
    # do augmentation
    img_out = adjustBrightness(image2d, as.integer(brightness_adjustments[a,1]), max_value)
    
    # reshape and store augmentation
    augmented_images[idx*num_augmentations+a,] = matrix(t(img_out), 1, width * height, byrow=TRUE)
  }
}

writeMM(as(augmented_images, "CsparseMatrix"), paste(args[2], "B", sep=""))

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

reverseMatrix = function(M) {
   out = apply(M, 2, rev)
}

images = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
width = as.integer(args[3])
height = as.integer(args[4])
augmented_images = matrix(0, 2 * nrow(images), ncol(images))

for (idx in 0:(nrow(images)-1)) {
  i = idx + 1
  
  image2d = matrix(images[i,], width, height)
  
  for(a in 1:2) {
    # do augmentation
    x_flip = t(reverseMatrix(t(image2d)))
    y_flip = reverseMatrix(image2d)
    
    # reshape and store augmentation
    augmented_images[idx*2+1,] = matrix(x_flip, 1, width * height)
    augmented_images[idx*2+2,] = matrix(y_flip, 1, width * height)
  }
}

writeMM(as(augmented_images, "CsparseMatrix"), paste(args[2], "B", sep=""))

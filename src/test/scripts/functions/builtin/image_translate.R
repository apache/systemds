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

image_translate = function(img_in, offset_x, offset_y) {
  width = ncol(img_in)
  height = nrow(img_in)
  img_out = matrix(0, nrow=height, ncol=width)
  for (x in 1:width) {
    for (y in 1:height) {
      src.x = x - round(offset_x)
      src.y = y - round(offset_y)
      if (0 < src.x & src.x <= width & 0 < src.y & src.y <= height) {
        img_out[x, y] = img_in[src.x, src.y]
      }
    }
  }

  img_out
}

input = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
input = matrix(input, as.integer(args[3]), as.integer(args[4]))

transformed = image_translate(input, as.double(args[5]), as.double(args[6]));
writeMM(as(transformed, "CsparseMatrix"), paste(args[2], "B", sep=""))

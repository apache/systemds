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

image_transform = function(img_in, out_w, out_h, a, b, c, d, e, f, fill_value) {
  divisor = a * e - b * d
  if (divisor == 0) {
    print("Inverse matrix does not exist! Returning input.")
    img_out = img_in
  }
  else {
    orig_w = ncol(img_in)
    orig_h = nrow(img_in)
    T.inv = matrix(0, nrow=3, ncol=3)
    T.inv[1, 1] = e / divisor
    T.inv[1, 2] = -b / divisor
    T.inv[1, 3] = (b * f - c * e) / divisor
    T.inv[2, 1] = -d / divisor
    T.inv[2, 2] = a / divisor
    T.inv[2, 3] = (c * d - a * f) / divisor
    T.inv[3, 3] = 1

    img_out = matrix(fill_value, nrow=out_w*out_h, ncol=1)
    coords = matrix(1, nrow=3, ncol=out_w*out_h)
    coords[1,] = t((seq(0, out_w*out_h-1) %% out_w) + 0.5)
    coords[2,] = t((seq(0, out_w*out_h-1) %/% out_w) + 0.5)
    coords = floor(T.inv %*% coords) + 1

    for (cell in 1:(out_w*out_h)) {
      inx = coords[1, cell]
      iny = coords[2, cell]
      if ((0 < inx) & (inx <= orig_w) & (0 < iny) & (iny <= orig_h)) {
        img_out[cell] = img_in[iny, inx]
      }
    }
    img_out = matrix(img_out, nrow=out_h, ncol=out_w, byrow=TRUE)
  }

  img_out
}

input = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
input = matrix(input, ncol=as.integer(args[3]), nrow=as.integer(args[4]))

transformed = image_transform(input, as.integer(args[5]), as.integer(args[6]), as.double(args[7]), as.double(args[8]), as.double(args[9]), as.double(args[10]), as.double(args[11]), as.double(args[12]), 0);
writeMM(as(transformed, "CsparseMatrix"), paste(args[2], "B", sep=""))

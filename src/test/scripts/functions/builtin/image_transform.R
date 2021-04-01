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

image_transform = function(img_in,  out_w,  out_h,  a, b, c, d, e, f) {
  print(a)
  print(b)
  print(c)
  print(d)
  print(e)
  print(f)
  divisor = a * e - b * d
  if(divisor == 0) {
    print("Inverse matrix does not exist! Returning input.")
    img_out = img_in
  }
  else {
    orig_w = ncol(img_in)
    orig_h = nrow(img_in)
    T.inv = matrix(c(e / divisor, -b / divisor, (b * f - c * e) / divisor,
                     -d / divisor, a / divisor, (c * d - a * f) / divisor,
                     0, 0, 1), nrow=3, ncol=3)

    img_out = matrix(nrow=out_h, ncol=out_w)
    for (x in 1:out_w) {
      for (y in 1:out_h) {
        coords = T.inv %*% matrix(c(x - 1, y - 1, 1), nrow=3)
        src.x = round(coords[1]) + 1
        src.y = round(coords[2]) + 1
        if (0 < src.x & src.x <= orig_w & 0 < src.y & src.y <= orig_h) {
          img_out[x, y] = img_in[src.x, src.y]
        }
      }
    }
  }

  img_out
}

input = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
input = matrix(input, as.integer(args[3]), as.integer(args[4]))

transformed = image_transform(input, as.integer(args[5]), as.integer(args[6]), as.integer(args[7]), as.integer(args[8]), as.integer(args[9]), as.integer(args[10]), as.integer(args[11]), as.integer(args[12]));
writeMM(as(transformed, "CsparseMatrix"), paste(args[2], "B", sep=""))

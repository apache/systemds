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

# wrapper for image_crop that takes in 2d matrix of linearized images and returns a 2d matrix of linearized cropped images
image_crop = function(img_in,  w,  h,  x_offset,  y_offset) {
  orig_w = ncol(img_in)
  orig_h = nrow(img_in)
  
  start_h = (ceiling((orig_h - h) / 2)) + y_offset
  end_h = (start_h + h - 1) 
  start_w = (ceiling((orig_w - w) / 2)) + x_offset
  end_w = (start_w + w - 1) 
  
  if((start_h < 0) | (end_h > orig_h) | (start_w < 0) | (end_w > orig_w)) {
  
    print("Offset out of bounds! Returning input.")
    img_out = img_in
  }
  else {
    mask = matrix(0, orig_h, orig_w)
    temp_mask = matrix(1, h , w )
    mask[start_h:end_h, start_w:end_w] = temp_mask
    mask = matrix(mask, 1, orig_w * orig_h)
    img_out = input[start_h:end_h, start_w:end_w]
  }

    return(img_out)
}

image_crop_linearized = function(img_in, new_w, new_h , s_cols, s_rows, x_offset, y_offset) {
  n_imgs = nrow(img_in)

  img_out = matrix(0,  n_imgs, new_w * new_h)

  
  for (i in 1:n_imgs) {
    row_in = matrix(img_in[i],  s_rows, s_cols)
    
    cropped_img = image_crop(row_in,  new_w, new_h, x_offset, y_offset)
    #cropped_img = image_crop(img_in, w, h, x_offset, y_offset)

    img_out[i] = matrix(cropped_img,  1, new_w * new_h)
  }
  
  return(img_out)
}


input = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))

input = matrix(input, as.integer(args[4]), as.integer(args[5]))
new_w = as.integer(args[10])
new_h = as.integer(args[11])
print(paste("R New w/h=",new_w,"/",new_h))
crop2 = image_crop_linearized(input, new_w, new_h, as.integer(args[6]), as.integer(args[7]), as.integer(args[8]), as.integer(args[9]));
writeMM(as(crop2, "CsparseMatrix"), paste(args[2], "B", sep=""))

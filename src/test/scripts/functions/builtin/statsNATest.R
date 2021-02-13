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
library("imputeTS")

input_matrix = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
input_matrix[input_matrix==0] = NA

bins_in = as.numeric(args[2])
output = matrix(0, nrow=8, ncol=1)

Out = statsNA(input_matrix, bins = bins_in, print_only = FALSE)

output[1,1]=Out["length_series"][[1]]
output[2,1]=Out["number_NAs"][[1]]
output[3,1]=as.numeric(sub("%","",Out["percentage_NAs"][[1]],fixed=TRUE))/100
output[4,1]=Out["number_na_gaps"][[1]]
output[5,1]=Out["average_size_na_gaps"][[1]]
output[6,1]=Out["longest_na_gap"][[1]]
output[7,1]=Out["most_frequent_na_gap"][[1]]
output[8,1]=Out["most_weighty_na_gap"][[1]]

writeMM(as(output, "CsparseMatrix"), paste(args[3], "Out", sep=""))


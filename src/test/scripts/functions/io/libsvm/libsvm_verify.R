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
options(digits=22)

filename <- args[1]
dimensionality <- as.integer(args[2])
sep <- args[3]
indSep <- args[4]

if(sep == 'NULL'){
  sep=" "
}

content = readLines( filename )
num_lines = length( content )
A = matrix( 0, num_lines, dimensionality + 1 )

# loop over lines
for ( i in 1:num_lines ) {
  # split by sep
  line = as.vector( strsplit( content[i], sep )[[1]])
  # save label
  A[i,1] = as.numeric( line[[1]] )
  # loop over values
  for ( j in 2:length( line )) {
    # split by colon
    index_value = strsplit( line[j], indSep )[[1]]
    index = as.numeric( index_value[1] ) + 1
     value = as.numeric( index_value[2] )
     A[i, index] = value
     }
  }
A[is.na(A)] = 0
A <- A+1
x =  sum(A)
write(x, args[5])
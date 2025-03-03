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

# Set options for numeric precision
options(digits=22)

# Load required libraries
library("Matrix")
library("matrixStats")

# Read operation type
type = as.integer(args[2])

# Create variables
X = matrix(1,1,1)
Y = matrix(2,1,1)
s = 2

# Perform the operations
if(type==1){
    R = as.numeric(X*Y)
} else if(type==2){
    R = as.numeric(X*s)
} else if(type==3){
    R = as.numeric(s*X)
}


write(R, paste(args[3], "R" ,sep=""))

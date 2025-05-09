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

# Read matrices and operation type
X = as.matrix(readMM(paste(args[1], "X.mtx", sep="")))
Y = as.matrix(readMM(paste(args[1], "Y.mtx", sep="")))
type = as.integer(args[2])

# Perform operations
# (1) Less
if(type==1){
    R = abs(X<Y)
} else if(type==2){
    R = round(X<Y)
} else if(type==3){
    R = ceiling(X<Y)
} else if(type==4){
    R = floor(X<Y)
} else if(type==5){
    R = sign(X<Y)
} else if(type==6){ # (2) Less-Equal
    R = abs(X<=Y)
} else if(type==7){
    R = round(X<=Y)
} else if(type==8){
    R = ceiling(X<=Y)
} else if(type==9){
    R = floor(X<=Y)
} else if(type==10){
    R = sign(X<=Y)
} else if(type==11){ # (3) Greater
    R = abs(X>Y)
} else if(type==12){
    R = round(X>Y)
} else if(type==13){
    R = ceiling(X>Y)
} else if(type==14){
    R = floor(X>Y)
} else if(type==15){
    R = sign(X>Y)
} else if(type==16){ # (4) Greater-Equal
    R = abs(X>=Y)
} else if(type==17){
    R = round(X>=Y)
} else if(type==18){
    R = ceiling(X>=Y)
} else if(type==19){
    R = floor(X>=Y)
} else if(type==20){
    R = sign(X>=Y)
} else if(type==21){ # (5) Equal
    R = abs(X==Y)
} else if(type==22){
    R = round(X==Y)
} else if(type==23){
    R = ceiling(X==Y)
} else if(type==24){
    R = floor(X==Y)
} else if(type==25){
    R = sign(X==Y)
} else if(type==26){ # (6) Not-Equal
    R = abs(X!=Y)
} else if(type==27){
    R = round(X!=Y)
} else if(type==28){
    R = ceiling(X!=Y)
} else if(type==29){
    R = floor(X!=Y)
} else if(type==30){
    R = sign(X!=Y)
}

#Write result matrix R
writeMM(as(R, "CsparseMatrix"), paste(args[3], "R", sep=""));

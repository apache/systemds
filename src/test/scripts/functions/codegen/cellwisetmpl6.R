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

args<-commandArgs(TRUE)
options(digits=22)
library("Matrix")

X = matrix(  c(1,2,3), nrow=3, ncol=1, byrow = TRUE)
y = matrix(  c(1,1,1), nrow=3, ncol=1, byrow = TRUE)
z = matrix(  c(3,3,3), nrow=3, ncol=1, byrow = TRUE)

S = sum(X*y*z)
print(S)
write(S,paste(args[2],"S",sep=""))

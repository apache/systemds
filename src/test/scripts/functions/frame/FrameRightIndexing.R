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
options(digits=22)
library("Matrix")


A=read.csv(paste(args[1], "A.csv", sep=""), header = FALSE, stringsAsFactors=FALSE)

B=A[args[2]:args[3],args[4]:args[5]]
C=A[1:args[3],args[4]:ncol(A)]
D=A[,args[4]:args[5]]

write.csv(B, paste(args[6], "B.csv", sep=""), row.names = FALSE, quote = FALSE)
write.csv(C, paste(args[6], "C.csv", sep=""), row.names = FALSE, quote = FALSE)
write.csv(D, paste(args[6], "D.csv", sep=""), row.names = FALSE, quote = FALSE)

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

ACsv=read.csv(paste(args[1], "ACsv", sep=""), header = FALSE, stringsAsFactors=FALSE)
BCsv=read.csv(paste(args[1], "BCsv", sep=""), header = FALSE, stringsAsFactors=FALSE)
CCsv=read.csv(paste(args[1], "CCsv", sep=""), header = FALSE, stringsAsFactors=FALSE)
DCsv=read.csv(paste(args[1], "DCsv", sep=""), header = FALSE, stringsAsFactors=FALSE)

ACsv[args[2]:args[3],args[4]:args[5]]=0
ACsv[args[2]:args[3],args[4]:args[5]]=BCsv
write.csv(ACsv, paste(args[6], "ABCsv", sep=""), row.names = FALSE, quote = FALSE)

ACsv[1:args[3],args[4]:ncol(ACsv)]=0
ACsv[1:args[3],args[4]:ncol(ACsv)]=CCsv
write.csv(ACsv, paste(args[6], "ACCsv", sep=""), row.names = FALSE, quote = FALSE)

ACsv[,args[4]:args[5]]=0
ACsv[,args[4]:args[5]]=DCsv
write.csv(ACsv, paste(args[6], "ADCsv", sep=""), row.names = FALSE, quote = FALSE)

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
library(Matrix)

t1 = as.numeric(args[1])
t2 = as.numeric(args[2])
t3 = args[3]

p  = pbinom(q=t2, size=20, prob=0.25, lower.tail=TRUE)
pl = pbinom(q=t2, size=20, prob=0.25, lower.tail=FALSE)
q  = qbinom(p=t1, size=20, prob=0.25)

res = rbind(as.matrix(p), as.matrix(pl), as.matrix(as.double(q)))

writeMM(as(res, "CsparseMatrix"), t3)

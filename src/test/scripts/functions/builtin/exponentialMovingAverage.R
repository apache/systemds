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

args <- commandArgs(trailingOnly = TRUE);
library("Matrix");
options(digits=22);

X = as.matrix(read.csv(paste(args[1], "F.csv", sep=""), header=FALSE));
output_path <- args[2]
mode = args[2+1];
freq = as.integer(args[3+1])
beta = FALSE;
gamma = FALSE;
alpha = as.double(args[3+2]);
if(mode == "double") {
  beta = as.double(args[4+2]);
} else {
  if(mode == "triple") {
    beta = as.double(args[4+2]);
    gamma = as.double(args[5+2]);
  }
}

ema_model = stats::HoltWinters(x=ts(X, frequency=freq), alpha=alpha, beta=beta, gamma=gamma);
R = as.numeric(ema_model$fitted[, "xhat"]);
writeMM(as(R, "CsparseMatrix"), paste(output_path, "O", sep=""));

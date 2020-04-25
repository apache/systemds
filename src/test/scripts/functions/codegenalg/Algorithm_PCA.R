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

#
# This script performs Principal Component Analysis (PCA) on the given input data.
#

args <- commandArgs(TRUE)
library("Matrix")

A = readMM(paste(args[1], "A.mtx", sep=""));
K = ncol(A);
projectData = 0;
model = "";
center = 0;
scale = 0;


if (model != "") {
  # reuse existing model to project data
} else if (model == "") {

  N = nrow(A);
  D = ncol(A);

  # 1. perform z-scoring (centering and scaling)
  if (center == 1) {
    cm = matrix(1, nrow(A), 1) %*% colMeans(A);
    A = A - cm
  }
  if (scale == 1) {
    cvars = (colSums(A^2));
    if (center == 1){
      #cm = colMeans(A);
      cvars = (cvars - N*(colMeans(A)^2))/(N-1);
    }
    Azscored = A / sqrt(cvars);
    A = Azscored;
  }

  # 2. compute co-variance matrix
  mu = colSums(A)/N;
  C = (t(A) %*% A)/(N-1) - (N/(N-1))*(mu) %*% t(mu);

  # 3. compute eigen vectors and values
  R <- eigen(C);
  evalues = R$values;
  evectors = R$vectors;

  # 4. make an index of values sorted according to magnitude of evalues
  decreasing_Idx = order(as.vector(evalues), decreasing=TRUE);
  diagmat = table(seq(1,D), decreasing_Idx);
  # 5. sorts eigen values by decreasing order
  evalues = diagmat %*% evalues;
  # 6. sorts eigen vectors column-wise in the order of decreasing eigen values
  evectors = evectors %*% diagmat;

  # 7. select K dominant eigen vectors
  nvec = ncol(evectors); # Here `nvec=K`
  eval_dominant = evalues[1:K, 1];
  evec_dominant = evectors[1:K,];

  # 8. compute the std. deviation of dominant evalues
  eval_stdev_dominant = sqrt(eval_dominant);

  writeMM(as(eval_stdev_dominant, "CsparseMatrix"), paste(args[2],"dominant.eigen.standard.deviations", sep=""));
  writeMM(as(eval_dominant, "CsparseMatrix"), paste(args[2], "dominant.eigen.values", sep=""));
  writeMM(as(evec_dominant, "CsparseMatrix"), paste(args[2],"dominant.eigen.vectors", sep=""));
}
#-------------------------------------------------------------
#
# (C) Copyright IBM Corp. 2010, 2015
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#-------------------------------------------------------------

# Intended to solve cubic natural spline, using R, in order to compare against the DML implementation
# INPUT 1: Matrix X [rows, 1]
# INPUT 2: Matrix Y [rows, 1]
# OUTPUT : 1x1 matrix of value of the interpolated y for given input x
#
# Assume that $CSPLINE_HOME is set to the home of the R script
# Assume input and output directories are $CSPLINE_HOME/in/ and $CSPLINE_HOME/expected/
# Rscript $CSPLINE_HOME/CsplineDs.R $CSPLINE_HOME/in/X.mtx $CSPLINE_HOME/in/Y.mtx 4.5 $CSPLINE_HOME/expected/y.mtx

args <- commandArgs (TRUE);

library ("Matrix");

X_here <- readMM (args[1]);  # (paste (args[1], "X.mtx", sep=""));
Y_here <- readMM (args[2]);  # (paste (args[2], "Y.mtx", sep=""));
inp_x <- args[3]
pred_y_here <- args[4]

X_matrix = as.matrix (X_here);
Y_matrix = as.matrix (Y_here);

sf<-splinefun(X_matrix, Y_matrix, method="natural")
pred_y = sf(inp_x)

print(paste("For inp_x = ", inp_x, " Calculated y = ", pred_y))

#print (c("Deviance", glmOut$deviance));
writeMM(as(pred_y, "CsparseMatrix") , pred_y_here);



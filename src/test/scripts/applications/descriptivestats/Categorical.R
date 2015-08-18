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

# JUnit test class: dml.test.integration.descriptivestats.UnivariateStatsTest.java
# command line invocation assuming $C_HOME is set to the home of the R script
# Rscript $C_HOME/Categorical.R $C_HOME/in/ $C_HOME/expected/
args <- commandArgs(TRUE)
options(digits=22)

#library("batch")
library("Matrix")

V = readMM(paste(args[1], "vector.mtx", sep=""))

tab = table(V[,1])
cat = t(as.numeric(names(tab)))
Nc = t(as.vector(tab))

# the number of categories of a categorical variable
R = length(Nc)

# total count
s = sum(Nc)

# percentage values of each categorical compare to the total case number
Pc = Nc / s

# all categorical values of a categorical variable
C = (Nc > 0)

# mode
mx = max(Nc)
Mode = (Nc == mx)

writeMM(as(t(Nc),"CsparseMatrix"), paste(args[2], "Nc", sep=""), format="text");
write(R, paste(args[2], "R", sep=""));
writeMM(as(t(Pc),"CsparseMatrix"), paste(args[2], "Pc", sep=""), format="text");
writeMM(as(t(C),"CsparseMatrix"), paste(args[2], "C", sep=""), format="text");
writeMM(as(t(Mode),"CsparseMatrix"), paste(args[2], "Mode", sep=""), format="text");

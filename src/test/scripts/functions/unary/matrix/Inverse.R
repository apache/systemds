#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2015
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------


args <- commandArgs(TRUE)
library(Matrix)

# Rscript ./test/scripts/functions/unary/matrix/Inverse.R ./test/scripts/functions/unary/matrix/in/A.mtx ./test/scripts/functions/unary/matrix/expected/AI

A = readMM(args[1]); 

AI = solve(A);

writeMM(as(AI, "CsparseMatrix"), args[2]); 


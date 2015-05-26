#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2015
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

A <- as.vector(readMM(paste(args[1], "A.mtx", sep="")))
B <- as.vector(readMM(paste(args[1], "B.mtx", sep="")))

opcode = args[2];
if( opcode == "lt" ) { opcode = "<" }
if( opcode == "mult" ) { opcode = "*" }

C <- outer(A, B, opcode)
C <- as.matrix(C)

writeMM(as(C, "CsparseMatrix"), paste(args[3], "C", sep="")); 



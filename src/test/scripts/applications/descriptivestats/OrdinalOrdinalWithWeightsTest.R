#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

# JUnit test class: dml.test.integration.descriptivestats.BivariateOrdinalOrdinalWithWeightsTest.java
# command line invocation assuming $OO_HOME is set to the home of the R script
# Rscript $OO_HOME/OrdinalOrdinal.R $OO_HOME/in/ $OO_HOME/expected/
args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

Atemp  = readMM(paste(args[1], "A.mtx", sep=""))
Btemp  = readMM(paste(args[1], "B.mtx", sep=""))
WMtemp = readMM(paste(args[1], "WM.mtx", sep=""))

#Atemp  = readMM(file="$$indir$$A.mtx"); #readMM(paste(args[1], "A.mtx", sep=""))
#Btemp  = readMM(file="$$indir$$B.mtx"); #readMM(paste(args[1], "B.mtx", sep=""))
#WMtemp = readMM(file="$$indir$$WM.mtx"); #readMM(paste(args[1], "WM.mtx", sep=""))

A = rep(Atemp[,1],WMtemp[,1])
B = rep(Btemp[,1],WMtemp[,1])

spearman = cor(A, B, method="spearman");

#paste("Weighted R value", spearman);

write(spearman, paste(args[2], "Spearman", sep=""));


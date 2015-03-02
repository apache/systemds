#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------


args <- commandArgs(TRUE)
options(digits=22)

library(Matrix);

# R interprets "0" as a categorical ("factor") value, so we need to read the
# file in as strings and convert everything to numeric explicitly.
A = read.csv(args[1], header=FALSE, stringsAsFactors=FALSE);
A = sapply(A, as.numeric);
x = sum(A);
write(x, args[2]);


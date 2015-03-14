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

qtle = qexp(as.numeric(args[1]), rate=as.numeric(args[2]));
p = pexp(qtle, rate=as.numeric(args[2]));
pl = pexp(qtle, rate=as.numeric(args[2]), lower.tail=F);

out = matrix(0,nrow=3, ncol=1);
out[1,1] = qtle;
out[2,1] = p;
out[3,1] = pl;

writeMM(as(out, "CsparseMatrix"), args[3]); 


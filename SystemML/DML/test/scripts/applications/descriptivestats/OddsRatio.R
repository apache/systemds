#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

# JUnit test class: dml.test.integration.descriptivestats.CategoricalCategoricalTest.java
# command line invocation assuming $CC_HOME is set to the home of the R script
# Rscript $CC_HOME/OddsRato.R $CC_HOME/in/ $CC_HOME/expected/

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

A = readMM(paste(args[1], "A.mtx", sep=""));
B = readMM(paste(args[1], "B.mtx", sep=""));

F = table(A[,1],B[,1]);

a11 = F[1,1];
a12 = F[1,2];
a21 = F[2,1];
a22 = F[2,2];

#print(paste(a11, " ", a12, " ", a21, " ", a22));

oddsRatio = (a11*a22)/(a12*a21);
sigma = sqrt(1/a11 + 1/a12 + 1/a21 + 1/a22);
left_conf = exp( log(oddsRatio) - 2*sigma )
right_conf = exp( log(oddsRatio) + 2*sigma )
sigma_away = abs( log(oddsRatio)/sigma )

# chisq.test returns a list containing statistic, p-value, etc.
cst = chisq.test(F);

# get the chi-squared coefficient from the list
chi_squared = as.numeric(cst[1]);
degFreedom =  as.numeric(cst[2]);
pValue = as.numeric(cst[3]);

q = min(dim(F));
W = sum(F);
cramers_v = sqrt(chi_squared/(W*(q-1)));


#print(paste(oddsRatio, " ", sigma, " [", left_conf, ",", right_conf, "] ", sigma_away, " "));
#print(paste(chi_squared, " ", degFreedom, " [", pValue, ",", cramers_v, "] "));

write(oddsRatio, paste(args[2], "oddsRatio", sep=""));
write(sigma, paste(args[2], "sigma", sep=""));
write(left_conf, paste(args[2], "leftConf", sep=""));
write(right_conf, paste(args[2], "rightConf", sep=""));
write(sigma_away, paste(args[2], "sigmasAway", sep=""));

#write(chi_squared, paste(args[2], "chiSquared", sep=""));
#write(degFreedom, paste(args[2], "degFreedom", sep=""));
#write(pValue, paste(args[2], "pValue", sep=""));
#write(cramers_v, paste(args[2], "cramersV", sep=""));


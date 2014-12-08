#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

# JUnit test class: dml.test.integration.descriptivestats.UnivariateStatsTest.java
# command line invocation assuming $S_HOME is set to the home of the R script
# Rscript $S_HOME/WeightedScaleTest.R $S_HOME/in/ $S_HOME/expected/
args <- commandArgs(TRUE)
options(digits=22)

options(repos="http://cran.stat.ucla.edu/") 
is.installed <- function(mypkg) is.element(mypkg, installed.packages()[,1])

is_plotrix = is.installed("plotrix");
if ( !is_plotrix ) {
install.packages("plotrix");
} 
library("plotrix");

is_psych = is.installed("psych");
if ( !is_psych ) {
install.packages("psych");
} 
library("psych")

is_moments = is.installed("moments");
if( !is_moments){
install.packages("moments");
}
library("moments")

#library("batch")
library("Matrix")

# Usage: R --vanilla -args Xfile X < DescriptiveStatistics.R

#parseCommandArgs()
######################

Temp = readMM(paste(args[1], "vector.mtx", sep=""))
W = readMM(paste(args[1], "weight.mtx", sep=""))
P = readMM(paste(args[1], "prob.mtx", sep=""))

W = round(W)

V=rep(Temp[,1],W[,1])

n = sum(W)

# sum
s1 = sum(V)

# mean
mu = s1/n

# variances
var = var(V)

# standard deviations
std_dev = sd(V, na.rm = FALSE)

# standard errors of mean
SE = std.error(V, na.rm)

# coefficients of variation
cv = std_dev/mu

# harmonic means (note: may generate out of memory for large sparse matrices becauses of NaNs)
har_mu = harmonic.mean(V)

# geometric means is not currently supported.
geom_mu = geometric.mean(V)

# min and max
mn=min(V)
mx=max(V)

# range
rng = mx - mn

# Skewness
g1 = n^2*moment(V, order=3, central=TRUE)/((n-1)*(n-2)*std_dev^3)

# standard error of skewness (not sure how it is defined without the weight)
se_g1=sqrt( 6*n*(n-1.0) / ((n-2.0)*(n+1.0)*(n+3.0)) )

m2 = moment(V, order=2, central=TRUE)
m4 = moment(V, order=4, central=TRUE)

# Kurtosis (using binomial formula)
g2 = (n^2*(n+1)*m4-3*m2^2*n^2*(n-1))/((n-1)*(n-2)*(n-3)*var^2)

# Standard error of Kurtosis (not sure how it is defined without the weight)
se_g2= sqrt( (4*(n^2-1)*se_g1^2)/((n+5)*(n-3)) )

# median
md = median(V) #quantile(V, 0.5, type = 1)

# quantile
Q = t(quantile(V, P[,1], type = 1))

# inter-quartile mean
S=c(sort(V))
n25=ceiling(length(S)*0.25)
n75=ceiling(length(S)*0.75)
T=S[(n25+1):n75]
iqm=mean(T)

# outliers use ppred to describe it
out_minus = t(as.numeric(Temp < mu-5*std_dev)*Temp) 
out_plus = t(as.numeric(Temp > mu+5*std_dev)*Temp)

write(mu, paste(args[2], "mean", sep=""));
write(std_dev, paste(args[2], "std", sep=""));
write(SE, paste(args[2], "se", sep=""));
write(var, paste(args[2], "var", sep=""));
write(cv, paste(args[2], "cv", sep=""));
# write(har_mu),paste(args[2], "har", sep=""));
# write(geom_mu, paste(args[2], "geom", sep=""));
write(mn, paste(args[2], "min", sep=""));
write(mx, paste(args[2], "max", sep=""));
write(rng, paste(args[2], "rng", sep=""));
write(g1, paste(args[2], "g1", sep=""));
write(se_g1, paste(args[2], "se_g1", sep=""));
write(g2, paste(args[2], "g2", sep=""));
write(se_g2, paste(args[2], "se_g2", sep=""));
write(md, paste(args[2], "median", sep=""));
write(iqm, paste(args[2], "iqm", sep=""));
writeMM(as(t(out_minus),"CsparseMatrix"), paste(args[2], "out_minus", sep=""), format="text");
writeMM(as(t(out_plus),"CsparseMatrix"), paste(args[2], "out_plus", sep=""), format="text");
writeMM(as(t(Q),"CsparseMatrix"), paste(args[2], "quantile", sep=""), format="text");

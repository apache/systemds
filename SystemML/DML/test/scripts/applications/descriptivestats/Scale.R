# JUnit test class: dml.test.integration.descriptivestats.UnivariateStatsTest.java
# command line invocation assuming $S_HOME is set to the home of the R script
# Rscript $S_HOME/Scale.R $S_HOME/in/ $S_HOME/expected/
args <- commandArgs(TRUE)

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

V = readMM(paste(args[1], "vector.mtx", sep=""))
P = readMM(paste(args[1], "prob.mtx", sep=""))

n = nrow(V)

# mean
mu = mean(V)

# variances
var = var(V[,1])

# standard deviations
std_dev = sd(V[,1], na.rm = FALSE)

# standard errors of mean
SE = std.error(V[,1], na.rm)

# coefficients of variation
cv = std_dev/mu

# harmonic means (note: may generate out of memory for large sparse matrices becauses of NaNs)
# har_mu = harmonic.mean(V[,1]) -- DML does not support this yet

# geometric means is not currently supported.
# geom_mu = geometric.mean(V[,1]) -- DML does not support this yet

# min and max
mn=min(V)
mx=max(V)

# range
rng = mx - mn

# Skewness
g1 = moment(V[,1], order=3, central=TRUE)/(std_dev^3)

# standard error of skewness (not sure how it is defined without the weight)
se_g1=sqrt( 6*n*(n-1.0) / ((n-2.0)*(n+1.0)*(n+3.0)) )

# Kurtosis (using binomial formula)
g2 = moment(V[,1], order=4, central=TRUE)/(var^2)-3

# Standard error of Kurtosis (not sure how it is defined without the weight)
se_g2= sqrt( (4*(n^2-1)*se_g1^2)/((n+5)*(n-3)) )

# median
md = quantile(V[,1], 0.5, type = 1)

# quantile
Q = t(quantile(V[,1], P[,1], type = 1))

# inter-quartile mean
S=c(sort(V[,1]))
n25=ceiling(length(S)*0.25)
n75=ceiling(length(S)*0.75)
T=S[(n25+1):n75]
iqm=mean(T)

# outliers use ppred to describe it
out_minus = t(as.numeric(V< mu-5*std_dev)*V) 
out_plus = t(as.numeric(V> mu+5*std_dev)*V)

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


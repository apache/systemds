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

V = readMM(file="$$indir$$vector.mtx")
P = readMM(file="$$indir$$prob.mtx")
Helper=matrix(1, 2, 1)

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
har_mu = harmonic.mean(V[,1])

# geometric means is not currently supported.
geom_mu = geometric.mean(V[,1])

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

meanHelper=mu*Helper
stdHelper=std_dev*Helper
SEHelper=SE*Helper
varHelper=var*Helper
cvHelper=cv*Helper
harHelper=har_mu*Helper
geomHelper=geom_mu*Helper
minHelper=mn*Helper
maxHelper=mx*Helper
rngHelper=rng*Helper
g1Helper=g1*Helper
se_g1Helper=se_g1*Helper
g2Helper=g2*Helper
se_g2Helper=se_g2*Helper
medianHelper1 = md * Helper
iqmHelper = iqm * Helper

writeMM(as(t(meanHelper),"CsparseMatrix"), "$$Routdir$$mean", format="text");
writeMM(as(t(stdHelper),"CsparseMatrix"), "$$Routdir$$std", format="text");
writeMM(as(t(SEHelper),"CsparseMatrix"), "$$Routdir$$se", format="text");
writeMM(as(t(varHelper),"CsparseMatrix"), "$$Routdir$$var", format="text");
writeMM(as(t(cvHelper),"CsparseMatrix"), "$$Routdir$$cv", format="text");
writeMM(as(t(harHelper),"CsparseMatrix"), "$$Routdir$$har", format="text");
writeMM(as(t(geomHelper),"CsparseMatrix"), "$$Routdir$$geom", format="text");
writeMM(as(t(minHelper),"CsparseMatrix"), "$$Routdir$$min", format="text");
writeMM(as(t(maxHelper),"CsparseMatrix"), "$$Routdir$$max", format="text");
writeMM(as(t(rngHelper),"CsparseMatrix"), "$$Routdir$$rng", format="text");
writeMM(as(t(g1Helper),"CsparseMatrix"), "$$Routdir$$g1", format="text");
writeMM(as(t(se_g1Helper),"CsparseMatrix"), "$$Routdir$$se_g1", format="text");
writeMM(as(t(g2Helper),"CsparseMatrix"), "$$Routdir$$g2", format="text");
writeMM(as(t(se_g2Helper),"CsparseMatrix"), "$$Routdir$$se_g2", format="text");
writeMM(as(t(out_minus),"CsparseMatrix"), "$$Routdir$$out_minus", format="text");
writeMM(as(t(out_plus),"CsparseMatrix"), "$$Routdir$$out_plus", format="text");
writeMM(as(t(medianHelper1),"CsparseMatrix"), "$$Routdir$$median", format="text");
writeMM(as(t(Q),"CsparseMatrix"), "$$Routdir$$quantile", format="text");
writeMM(as(t(iqmHelper),"CsparseMatrix"), "$$Routdir$$iqm", format="text");

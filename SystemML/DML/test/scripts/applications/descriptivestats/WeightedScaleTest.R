options(repos="http://cran.stat.ucla.edu/") 
is.installed <- function(mypkg) is.element(mypkg, installed.packages()[,1])

is_plotrix = is.installed("plotrix");
if ( !is_plotrix ) {
install.package("plotrix");
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

Temp = readMM(file="$$indir$$vector.mtx")
P = readMM(file="$$indir$$prob.mtx")
W = readMM(file="$$indir$$weight.mtx")
Helper=matrix(1, 2, 1)

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
md = quantile(V, 0.5, type = 1)

# quantile
Q = t(quantile(V, P[,1], type = 1))

# inter-quartile mean
S=c(sort(V))
n25=ceiling(length(S)*0.25)
n75=ceiling(length(S)*0.75)
T=S[(n25+1):n75]
iqm=mean(T)

# outliers use ppred to describe it
out_minus = as.numeric(Temp < mu-5*std_dev)*Temp 
out_plus = as.numeric(Temp > mu+5*std_dev)*Temp

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

writeMM(as(t(meanHelper),"CsparseMatrix"), "$$Routdir$$mean_weight", format="text");
writeMM(as(t(stdHelper),"CsparseMatrix"), "$$Routdir$$std_weight", format="text");
writeMM(as(t(SEHelper),"CsparseMatrix"), "$$Routdir$$se_weight", format="text");
writeMM(as(t(varHelper),"CsparseMatrix"), "$$Routdir$$var_weight", format="text");
writeMM(as(t(cvHelper),"CsparseMatrix"), "$$Routdir$$cv_weight", format="text");
writeMM(as(t(harHelper),"CsparseMatrix"), "$$Routdir$$har_weight", format="text");
writeMM(as(t(geomHelper),"CsparseMatrix"), "$$Routdir$$geom_weight", format="text");
writeMM(as(t(minHelper),"CsparseMatrix"), "$$Routdir$$min_weight", format="text");
writeMM(as(t(maxHelper),"CsparseMatrix"), "$$Routdir$$max_weight", format="text");
writeMM(as(t(rngHelper),"CsparseMatrix"), "$$Routdir$$rng_weight", format="text");
writeMM(as(t(g1Helper),"CsparseMatrix"), "$$Routdir$$g1_weight", format="text");
writeMM(as(t(se_g1Helper),"CsparseMatrix"), "$$Routdir$$se_g1_weight", format="text");
writeMM(as(t(g2Helper),"CsparseMatrix"), "$$Routdir$$g2_weight", format="text");
writeMM(as(t(se_g2Helper),"CsparseMatrix"), "$$Routdir$$se_g2_weight", format="text");
writeMM(as(t(out_minus),"CsparseMatrix"), "$$Routdir$$out_minus_weight", format="text");
writeMM(as(t(out_plus),"CsparseMatrix"), "$$Routdir$$out_plus_weight", format="text");
writeMM(as(t(medianHelper1),"CsparseMatrix"), "$$Routdir$$median_weight", format="text");
writeMM(as(t(Q),"CsparseMatrix"), "$$Routdir$$quantile_weight", format="text");
writeMM(as(t(iqmHelper),"CsparseMatrix"), "$$Routdir$$iqm_weight", format="text");
#-------------------------------------------------------------
#
# (C) Copyright IBM Corp. 2010, 2015
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#-------------------------------------------------------------

bivar_ss = function(X, Y) {

    # Unweighted co-variance
    covXY = cov(X,Y)

    # compute standard deviations for both X and Y by computing 2^nd central moment
    m2X = var(X)
    m2Y = var(Y)
    sigmaX = sqrt(m2X)
    sigmaY = sqrt(m2Y)

    # Pearson's R
    R = covXY / (sigmaX*sigmaY)

    return(list("R" = R, "covXY" = covXY, "sigmaX" = sigmaX, "sigmaY" = sigmaY))
}

# -----------------------------------------------------------------------------------------------------------

bivar_cc = function(A, B) {

    # Contingency Table
    F = table(A,B)
    
    # Chi-Squared
    cst = chisq.test(F)

    r = rowSums(F)
    c = colSums(F)
    
    chi_squared = as.numeric(cst[1])

    # compute p-value
    pValue = as.numeric(cst[3])

    # Assign return values
    pval = pValue
    contingencyTable = F
    rowMarginals = r
    colMarginals = c

    return(list("pval" = pval, "contingencyTable" = contingencyTable, "rowMarginals" = rowMarginals, "colMarginals" = colMarginals))
}

# -----------------------------------------------------------------------------------------------------------

# Y points to SCALE variable
# A points to CATEGORICAL variable
bivar_sc = function(Y, A) {
    # mean and variance in target variable
    W = length(A)
    my = mean(Y)
    varY = var(Y)

    # category-wise (frequencies, means, variances)
    CFreqs = as.matrix(table(A)) 

    CMeans = as.matrix(aggregate(Y, by=list(A), "mean")$x)

    CVars = as.matrix(aggregate(Y, by=list(A), "var")$x)
    CVars[is.na(CVars)] <- 0

    # number of categories
    R = nrow(CFreqs)
    df1 = R-1
    df2 = W-R

    anova_num = sum( (CFreqs*(CMeans-my)^2) )/(R-1)
    anova_den = sum( (CFreqs-1)*CVars )/(W-R)
    AnovaF = anova_num/anova_den
    pVal = 1-pf(AnovaF, df1, df2)

    return(list("pVal" = pVal, "CFreqs" = CFreqs, "CMeans" = CMeans, "CVars" = CVars))
}

# Main starts here -----------------------------------------------------------------------------------------------------------

args <- commandArgs(TRUE)

library(Matrix)

# input data set
D = readMM(paste(args[1], "X.mtx", sep=""));

# label attr id (must be a valid index > 0)  
label_index = as.integer(args[2])

# feature attributes, column vector of indices
feature_indices = readMM(paste(args[1], "feature_indices.mtx", sep="")) 

# can be either 1 (scale) or 0 (categorical)
label_measurement_level = as.integer(args[3]) 

# measurement levels for features, 0/1 column vector
feature_measurement_levels = readMM(paste(args[1], "feature_measurement_levels.mtx", sep="")) 

sz = ncol(D)

# store for pvalues and pearson's r
stats = matrix(0, sz, 1)
# store for type of test performed: 1 is chi-sq, 2 is ftest, 3 is pearson's
tests = matrix(0, sz, 1)
# store for covariances used to compute pearson's r
covariances = matrix(0, sz, 1)
# store for standard deviations used to compute pearson's r
standard_deviations = matrix(0, sz, 1)

labels = D[,label_index]

labelCorrection = 0
if(label_measurement_level == 1){
	numLabels = length(labels)
        cmLabels = var(labels)
    	stdLabels = sqrt(cmLabels)
	standard_deviations[label_index,1] = stdLabels
}else{
	labelCorrection = 1 - min(labels)
	labels = labels + labelCorrection
}

mx = apply(D, 2, max)
mn = apply(D, 2, min)	
num_distinct_values = mx-mn+1
max_num_distinct_values = 0
for(i1 in 1:nrow(feature_indices)){
	feature_index1 = feature_indices[i1,1]
	num = num_distinct_values[feature_index1]
	if(feature_measurement_levels[i1,1] == 0 & num >= max_num_distinct_values){
		max_num_distinct_values = num
	}
}
distinct_label_values = matrix(0, 1, 1)	
contingencyTableSz = 1
maxNumberOfGroups = 1
if(max_num_distinct_values != 0){
	maxNumberOfGroups = max_num_distinct_values
}
if(label_measurement_level==0){
	distinct_label_values = as.data.frame(table(labels))$Freq
	if(max_num_distinct_values != 0){
		contingencyTableSz = max_num_distinct_values*length(distinct_label_values)
	}
	maxNumberOfGroups = max(maxNumberOfGroups, length(distinct_label_values))
}
# store for contingency table cell values
contingencyTablesCounts = matrix(0, sz, contingencyTableSz)
# store for contingency table label(row) assignments
contingencyTablesLabelValues = matrix(0, sz, contingencyTableSz)
# store for contingency table feature(col) assignments
contingencyTablesFeatureValues = matrix(0, sz, contingencyTableSz)
# store for distinct values
featureValues = matrix(0, sz, maxNumberOfGroups)
# store for counts of distinct values
featureCounts = matrix(0, sz, maxNumberOfGroups)
# store for group means
featureMeans = matrix(0, sz, maxNumberOfGroups)
# store for group standard deviations
featureSTDs = matrix(0, sz, maxNumberOfGroups)

if(label_measurement_level == 0){
	featureCounts[label_index,1:length(distinct_label_values)] = distinct_label_values
	for(i2 in 1:length(distinct_label_values)){
		featureValues[label_index,i2] = i2-labelCorrection
	}
}

for(i3 in 1:nrow(feature_indices)){
	feature_index2 = feature_indices[i3,1]
	feature_measurement_level = feature_measurement_levels[i3,1]
	
	feature = D[,feature_index2]
	
	if(feature_measurement_level == 0){
		featureCorrection = 1 - min(feature)
		feature = feature + featureCorrection
			
		if(label_measurement_level == feature_measurement_level){
		  # categorical-categorical
		  tests[feature_index2,1] = 1

		  ret = bivar_cc(labels, feature)
                  pVal = ret$pval
                  contingencyTable = ret$contingencyTable
                  rowMarginals = ret$rowMarginals
                  colMarginals = ret$colMarginals

		  stats[feature_index2,1] = pVal
			
		  sz3 = nrow(contingencyTable)*ncol(contingencyTable)
			
		  contingencyTableCounts = matrix(0, 1, sz3)
		  contingencyTableLabelValues = matrix(0, 1, sz3)
		  contingencyTableFeatureValues = matrix(0, 1, sz3)
			
            	  for(i4 in 1:nrow(contingencyTable)){
		  	 for(j in 1:ncol(contingencyTable)){
					#get rid of this, see *1 below
					contingencyTableCounts[1, ncol(contingencyTable)*(i4-1)+j] = contingencyTable[i4,j]
					
					contingencyTableLabelValues[1, ncol(contingencyTable)*(i4-1)+j] = i4-labelCorrection
					contingencyTableFeatureValues[1, ncol(contingencyTable)*(i4-1)+j] = j-featureCorrection 
				}
			}
			contingencyTablesCounts[feature_index2,1:sz3] = contingencyTableCounts
            
			contingencyTablesLabelValues[feature_index2,1:sz3] = contingencyTableLabelValues
			contingencyTablesFeatureValues[feature_index2,1:sz3] = contingencyTableFeatureValues
			
			featureCounts[feature_index2,1:length(colMarginals)] = colMarginals
			for(i5 in 1:length(colMarginals)){
				featureValues[feature_index2,i5] = i5-featureCorrection
			}
		}else{
			# label is scale, feature is categorical
			tests[feature_index2,1] = 2
			
			ret = bivar_sc(labels, feature)
                  pVal = ret$pVal
                  frequencies = ret$CFreqs
                  means = ret$CMeans
                  variances = ret$CVars

			stats[feature_index2,1] = pVal
			featureCounts[feature_index2,1:nrow(frequencies)] = t(frequencies)
			for(i6 in 1:nrow(frequencies)){
				featureValues[feature_index2,i6] = i6 - featureCorrection
			}
			featureMeans[feature_index2,1:nrow(means)] = t(means)
			featureSTDs[feature_index2,1:nrow(variances)] = t(sqrt(variances))
		}
	}else{
		if(label_measurement_level == feature_measurement_level){
		  # scale-scale
		  tests[feature_index2,1] = 3

		  ret = bivar_ss(labels, feature)
                  r = ret$R
                  covariance = ret$covXY
                  stdX = ret$sigmaX
                  stdY = ret$sigmaY
 
		  stats[feature_index2,1] = r
		  covariances[feature_index2,1] = covariance
		  standard_deviations[feature_index2,1] = stdY
		}else{
		  # label is categorical, feature is scale
		  tests[feature_index2,1] = 2
			
		  ret = bivar_sc(feature, labels)
		  pVal = ret$pVal
		  frequencies = ret$CFreqs
                  means = ret$CMeans
		  variances = ret$CVars
			
		  stats[feature_index2,1] = pVal
		  featureMeans[feature_index2,1:nrow(means)] = t(means)
		  featureSTDs[feature_index2,1:nrow(variances)] = t(sqrt(variances))
		}
	}
}

writeMM(as(stats, "CsparseMatrix"), paste(args[4], "stats", sep=""))
writeMM(as(tests, "CsparseMatrix"), paste(args[4], "tests", sep=""))
writeMM(as(covariances, "CsparseMatrix"), paste(args[4], "covariances", sep=""))
writeMM(as(standard_deviations, "CsparseMatrix"), paste(args[4], "standard_deviations", sep=""))
writeMM(as(contingencyTablesCounts, "CsparseMatrix"), paste(args[4], "contingencyTablesCounts", sep=""))
writeMM(as(contingencyTablesLabelValues, "CsparseMatrix"), paste(args[4], "contingencyTablesLabelValues", sep=""))
writeMM(as(contingencyTablesFeatureValues, "CsparseMatrix"), paste(args[4], "contingencyTablesFeatureValues", sep=""))
writeMM(as(featureValues, "CsparseMatrix"), paste(args[4], "featureValues", sep=""))
writeMM(as(featureCounts, "CsparseMatrix"), paste(args[4], "featureCounts", sep=""))
writeMM(as(featureMeans, "CsparseMatrix"), paste(args[4], "featureMeans", sep=""))
writeMM(as(featureSTDs, "CsparseMatrix"), paste(args[4], "featureSTDs", sep=""))


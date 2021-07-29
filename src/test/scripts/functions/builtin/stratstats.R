#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------
library("Matrix")
args <- commandArgs(TRUE)

fStat_tailprob = function(fStat, df_1, df_2){
 rows = nrow(fStat)
 if(is.null(rows)){
   rows = 1
   fStat = t(matrix(fStat))
 }

 cols = ncol(fStat)
 if(is.null(cols))
    cols = length(fStat)

 if(is.null(nrow(df_1)))
    df_1 = t(matrix(df_1))

 if(is.null(nrow(df_2)))
    df_2 = t(matrix(df_2))

 tailprob = fStat
 for (i in 1:rows) {
   for (j in 1:cols) {
     q = fStat[i, j]
     d1 = df_1[i, j]
     d2 = df_2[i, j]
     if (d1 >= 1 & d2 >= 1 & q >= 0)
        tailprob[i, j] = pf(q, df1 = d1, df2 = d2, lower.tail=FALSE)
      else
        tailprob[i, j] = 0/0
    }
   }
   return(tailprob)
}

sqrt_failsafe = function(input_A){
 mask_A = (input_A >= 0)
 prep_A = input_A * mask_A
 mask_A = mask_A * (prep_A == prep_A)
 #prep_A = replace (target = prep_A, pattern = NaN, replacement = 0)
 output_A = sqrt (prep_A) / mask_A
 return(output_A)
}




stratum_column_id = 1
XwithNaNs = readMM(args[1])
YwithNaNs = XwithNaNs
SwithNaNs = XwithNaNs[, stratum_column_id]
Xcols = c(1:ncol(XwithNaNs))
Ycols = c(1:ncol(YwithNaNs))

num_records  = nrow(XwithNaNs)
num_attrs    = ncol(XwithNaNs)
num_attrs_X  = length(Xcols)
num_attrs_Y  = length(Ycols)
num_attrs_XY = num_attrs_X * num_attrs_Y

one_to_num_attrs_X = c(1:num_attrs_X)
one_to_num_attrs_Y = c(1:num_attrs_Y)
ProjX = matrix(0, nrow = num_attrs, ncol = num_attrs_X)
ProjY = matrix(0, nrow = num_attrs, ncol = num_attrs_Y)

ProjX_ctable = table(t(Xcols), one_to_num_attrs_X)
ProjX[1:nrow(ProjX_ctable), ] = ProjX_ctable
ProjY_ctable = table(t(Ycols), one_to_num_attrs_Y)
ProjY[1:nrow(ProjY_ctable), ] = ProjY_ctable

X = XwithNaNs %*% ProjX
Y = XwithNaNs %*% ProjY
S = round(SwithNaNs) * (SwithNaNs > 0)

Proj_good_stratumID = diag(S > 0)
Proj_good_stratumID = Proj_good_stratumID[rowSums(Proj_good_stratumID[])>0,]
vector_of_good_stratumIDs = Proj_good_stratumID %*% S
vector_of_good_stratumIDs = vector_of_good_stratumIDs + (1 - min(vector_of_good_stratumIDs))
num_records_with_good_stratumID = nrow(Proj_good_stratumID)
if (is.null(num_records_with_good_stratumID))
    num_records_with_good_stratumID = 1
one_to_num_records_with_good_stratumID = seq(1,num_records_with_good_stratumID,by=1)

num_strata_with_empty = max(vector_of_good_stratumIDs)
StrataSummator_with_empty = table(vector_of_good_stratumIDs, one_to_num_records_with_good_stratumID)
StrataSummator = StrataSummator_with_empty[rowSums(StrataSummator_with_empty[])>0,]
StrataSummator = StrataSummator %*% Proj_good_stratumID
num_strata = nrow(StrataSummator)
num_empty_strata = num_strata_with_empty - num_strata

XNaNmask = (XwithNaNs == XwithNaNs)
YNaNmask = (YwithNaNs == YwithNaNs)
X_mask = XNaNmask %*% ProjX
Y_mask = YNaNmask %*% ProjY

cnt_X_global = colSums(X_mask)
cnt_Y_global = colSums(Y_mask)
avg_X_global = colSums(X) / cnt_X_global
avg_Y_global = colSums(Y) / cnt_Y_global
var_sumX_global = colSums (X * X) - cnt_X_global * (avg_X_global * avg_X_global)
var_sumY_global = colSums (Y * Y) - cnt_Y_global * (avg_Y_global * avg_Y_global)
sqrt_failsafe_input_1 = var_sumX_global / (cnt_X_global - 1)
stdev_X_global = sqrt_failsafe (sqrt_failsafe_input_1)
sqrt_failsafe_input_2 = var_sumY_global / (cnt_Y_global - 1)
stdev_Y_global = sqrt_failsafe (sqrt_failsafe_input_2)

Cnt_X_per_stratum = StrataSummator %*% X_mask
Cnt_Y_per_stratum = StrataSummator %*% Y_mask
Is_none_X_per_stratum = (Cnt_X_per_stratum == 0)
Is_none_Y_per_stratum = (Cnt_Y_per_stratum == 0)
One_over_cnt_X_per_stratum = (1 - Is_none_X_per_stratum) / (Cnt_X_per_stratum + Is_none_X_per_stratum)
One_over_cnt_Y_per_stratum = (1 - Is_none_Y_per_stratum) / (Cnt_Y_per_stratum + Is_none_Y_per_stratum)
num_X_nonempty_strata = num_strata - colSums (Is_none_X_per_stratum)
num_Y_nonempty_strata = num_strata - colSums (Is_none_Y_per_stratum)

Sum_X_per_stratum  = StrataSummator %*% X
Sum_Y_per_stratum  = StrataSummator %*% Y

cnt_X_with_good_stratumID = colSums(Cnt_X_per_stratum)
cnt_Y_with_good_stratumID = colSums(Cnt_Y_per_stratum)
sum_X_with_good_stratumID = colSums(Sum_X_per_stratum)
sum_Y_with_good_stratumID = colSums(Sum_Y_per_stratum)
var_sumX_with_good_stratumID = colSums(StrataSummator %*% (X * X)) - (sum_X_with_good_stratumID * sum_X_with_good_stratumID) / cnt_X_with_good_stratumID
var_sumY_with_good_stratumID = colSums(StrataSummator %*% (Y * Y)) - (sum_Y_with_good_stratumID * sum_Y_with_good_stratumID) / cnt_Y_with_good_stratumID

var_sumX_stratified   = colSums (StrataSummator %*% (X * X)) - colSums (One_over_cnt_X_per_stratum * Sum_X_per_stratum * Sum_X_per_stratum)
var_sumY_stratified   = colSums (StrataSummator %*% (Y * Y)) - colSums (One_over_cnt_Y_per_stratum * Sum_Y_per_stratum * Sum_Y_per_stratum)
sqrt_failsafe_input_3 = var_sumX_stratified / (cnt_X_with_good_stratumID - num_X_nonempty_strata)
stdev_X_stratified    = sqrt_failsafe (sqrt_failsafe_input_3)
sqrt_failsafe_input_4 = var_sumY_stratified / (cnt_Y_with_good_stratumID - num_Y_nonempty_strata)
stdev_Y_stratified    = sqrt_failsafe (sqrt_failsafe_input_4)
r_sqr_X_vs_strata     = 1 - var_sumX_stratified / var_sumX_with_good_stratumID
r_sqr_Y_vs_strata     = 1 - var_sumY_stratified / var_sumY_with_good_stratumID
adj_r_sqr_X_vs_strata = 1 - (var_sumX_stratified / (cnt_X_with_good_stratumID - num_X_nonempty_strata)) / (var_sumX_with_good_stratumID / (cnt_X_with_good_stratumID - 1))
adj_r_sqr_Y_vs_strata = 1 - (var_sumY_stratified / (cnt_Y_with_good_stratumID - num_Y_nonempty_strata)) / (var_sumY_with_good_stratumID / (cnt_Y_with_good_stratumID - 1))
fStat_X_vs_strata     = ((var_sumX_with_good_stratumID - var_sumX_stratified) / (num_X_nonempty_strata - 1)) / (var_sumX_stratified / (cnt_X_with_good_stratumID - num_X_nonempty_strata))
fStat_Y_vs_strata     = ((var_sumY_with_good_stratumID - var_sumY_stratified) / (num_Y_nonempty_strata - 1)) / (var_sumY_stratified / (cnt_Y_with_good_stratumID - num_Y_nonempty_strata))
p_val_X_vs_strata     = fStat_tailprob (fStat_X_vs_strata, num_X_nonempty_strata - 1, cnt_X_with_good_stratumID - num_X_nonempty_strata)
p_val_Y_vs_strata     = fStat_tailprob (fStat_Y_vs_strata, num_Y_nonempty_strata - 1, cnt_Y_with_good_stratumID - num_Y_nonempty_strata)

cnt_XY_rectangle       = t(X_mask) %*% Y_mask
sum_X_forXY_rectangle  = t(X)      %*% Y_mask
sum_XX_forXY_rectangle = t(X * X)  %*% Y_mask
sum_Y_forXY_rectangle  = t(X_mask) %*% Y
sum_YY_forXY_rectangle = t(X_mask) %*% (Y * Y)
sum_XY_rectangle       = t(X)      %*% Y
cnt_XY_global       = matrix(t(cnt_XY_rectangle),       nrow = 1, ncol = num_attrs_XY)
sum_X_forXY_global  = matrix(t(sum_X_forXY_rectangle), nrow = 1, byrow = TRUE)
sum_XX_forXY_global = matrix(t(sum_XX_forXY_rectangle), nrow = 1, ncol = num_attrs_XY, byrow = TRUE)
sum_Y_forXY_global  = matrix(t(sum_Y_forXY_rectangle),  nrow = 1, ncol = num_attrs_XY, byrow = TRUE)
sum_YY_forXY_global = matrix(t(sum_YY_forXY_rectangle), nrow = 1, ncol = num_attrs_XY, byrow = TRUE)
sum_XY_global       = matrix(sum_XY_rectangle,       nrow = 1, ncol = num_attrs_XY, byrow = TRUE)
ones_XY = matrix(1.0, nrow = 1, ncol = num_attrs_XY)

cov_sumX_sumY_global    = sum_XY_global - sum_X_forXY_global * sum_Y_forXY_global / cnt_XY_global
var_sumX_forXY_global   = sum_XX_forXY_global - sum_X_forXY_global * sum_X_forXY_global / cnt_XY_global
var_sumY_forXY_global   = sum_YY_forXY_global - sum_Y_forXY_global * sum_Y_forXY_global / cnt_XY_global

slope_XY_global         = cov_sumX_sumY_global / var_sumX_forXY_global
sqrt_failsafe_input_5 = var_sumX_forXY_global * var_sumY_forXY_global
sqrt_failsafe_output_5 = sqrt_failsafe(sqrt_failsafe_input_5)
corr_XY_global          = cov_sumX_sumY_global / sqrt_failsafe_output_5
r_sqr_X_vs_Y_global     = cov_sumX_sumY_global * cov_sumX_sumY_global / (var_sumX_forXY_global * var_sumY_forXY_global)
adj_r_sqr_X_vs_Y_global = 1 - (1 - r_sqr_X_vs_Y_global) * (cnt_XY_global - 1) / (cnt_XY_global - 2)
sqrt_failsafe_input_6 = (1 - r_sqr_X_vs_Y_global) * var_sumY_forXY_global / var_sumX_forXY_global / (cnt_XY_global - 2)
stdev_slope_XY_global   = sqrt_failsafe(sqrt_failsafe_input_6)
sqrt_failsafe_input_7 = (1 - r_sqr_X_vs_Y_global) * var_sumY_forXY_global / (cnt_XY_global - 2)
stdev_errY_vs_X_global  = sqrt_failsafe(sqrt_failsafe_input_7)
fStat_Y_vs_X_global     = (cnt_XY_global - 2) * r_sqr_X_vs_Y_global / (1 - r_sqr_X_vs_Y_global)
p_val_Y_vs_X_global     = fStat_tailprob(fStat_Y_vs_X_global, ones_XY, cnt_XY_global - 2)

Proj_X_to_XY = matrix(0.0, nrow = num_attrs_X, ncol = num_attrs_XY)
Proj_Y_to_XY = matrix(0.0, nrow = num_attrs_Y, ncol = num_attrs_XY)
ones_Y_col   = matrix(1.0, nrow = num_attrs_Y, ncol = 1)
for (i in 1:num_attrs_X) {
    start_cid = (i - 1) * num_attrs_Y + 1
    end_cid = i * num_attrs_Y
    Proj_X_to_XY [i, start_cid:end_cid] = t(ones_Y_col)
    Proj_Y_to_XY [ , start_cid:end_cid] = diag(nrow=nrow(ones_Y_col), ncol=nrow(ones_Y_col))
}

Cnt_XY_per_stratum       = StrataSummator %*% (( X_mask %*% Proj_X_to_XY) * ( Y_mask %*% Proj_Y_to_XY))
Sum_X_forXY_per_stratum  = StrataSummator %*% (( X      %*% Proj_X_to_XY) * ( Y_mask %*% Proj_Y_to_XY))
Sum_XX_forXY_per_stratum = StrataSummator %*% (((X * X) %*% Proj_X_to_XY) * ( Y_mask %*% Proj_Y_to_XY))
Sum_Y_forXY_per_stratum  = StrataSummator %*% (( X_mask %*% Proj_X_to_XY) * ( Y      %*% Proj_Y_to_XY))
Sum_YY_forXY_per_stratum = StrataSummator %*% (( X_mask %*% Proj_X_to_XY) * ((Y * Y) %*% Proj_Y_to_XY))
Sum_XY_per_stratum       = StrataSummator %*% (( X      %*% Proj_X_to_XY) * ( Y      %*% Proj_Y_to_XY))

Is_none_XY_per_stratum = (Cnt_XY_per_stratum == 0)
One_over_cnt_XY_per_stratum = (1 - Is_none_XY_per_stratum) / (Cnt_XY_per_stratum + Is_none_XY_per_stratum)
num_XY_nonempty_strata = num_strata - colSums (Is_none_XY_per_stratum)

cnt_XY_with_good_stratumID = colSums(Cnt_XY_per_stratum)
sum_XX_forXY_with_good_stratumID = colSums(Sum_XX_forXY_per_stratum)
sum_YY_forXY_with_good_stratumID = colSums(Sum_YY_forXY_per_stratum)
sum_XY_with_good_stratumID = colSums(Sum_XY_per_stratum)

var_sumX_forXY_stratified = sum_XX_forXY_with_good_stratumID - colSums(Sum_X_forXY_per_stratum * Sum_X_forXY_per_stratum * One_over_cnt_XY_per_stratum)
var_sumY_forXY_stratified = sum_YY_forXY_with_good_stratumID - colSums(Sum_Y_forXY_per_stratum * Sum_Y_forXY_per_stratum * One_over_cnt_XY_per_stratum)
cov_sumX_sumY_stratified  = sum_XY_with_good_stratumID       - colSums(Sum_X_forXY_per_stratum * Sum_Y_forXY_per_stratum * One_over_cnt_XY_per_stratum)

slope_XY_stratified     = cov_sumX_sumY_stratified / var_sumX_forXY_stratified
sqrt_failsafe_input_8 = var_sumX_forXY_stratified * var_sumY_forXY_stratified
sqrt_failsafe_output_8 = sqrt_failsafe(sqrt_failsafe_input_8)
corr_XY_stratified      = cov_sumX_sumY_stratified / sqrt_failsafe_output_8
r_sqr_X_vs_Y_stratified = (cov_sumX_sumY_stratified ^ 2) / (var_sumX_forXY_stratified * var_sumY_forXY_stratified)
temp_X_vs_Y_stratified  = (1 - r_sqr_X_vs_Y_stratified) / (cnt_XY_with_good_stratumID - num_XY_nonempty_strata - 1)
adj_r_sqr_X_vs_Y_stratified = 1 - temp_X_vs_Y_stratified * (cnt_XY_with_good_stratumID - num_XY_nonempty_strata)
sqrt_failsafe_input_9  = temp_X_vs_Y_stratified * var_sumY_forXY_stratified

if(all(var_sumY_forXY_stratified == 0)){
  sqrt_failsafe_input_9 = var_sumY_forXY_stratified
}

stdev_errY_vs_X_stratified  = sqrt_failsafe (sqrt_failsafe_input_9)
sqrt_failsafe_input_10 = sqrt_failsafe_input_9  / var_sumX_forXY_stratified
stdev_slope_XY_stratified   = sqrt_failsafe(sqrt_failsafe_input_10)
#stdev_slope_XY_stratified[is.na(stdev_slope_XY_stratified)] = NaN
fStat_Y_vs_X_stratified = (cnt_XY_with_good_stratumID - num_XY_nonempty_strata - 1) * r_sqr_X_vs_Y_stratified / (1 - r_sqr_X_vs_Y_stratified)
p_val_Y_vs_X_stratified = fStat_tailprob (fStat_Y_vs_X_stratified, ones_XY, cnt_XY_with_good_stratumID - num_XY_nonempty_strata - 1)

OutMtx = matrix(0.0, nrow = 40, ncol = num_attrs_XY)
OutMtx [ 1, ] = Xcols                 %*% Proj_X_to_XY
OutMtx [ 2, ] = cnt_X_global          %*% Proj_X_to_XY
OutMtx [ 3, ] = avg_X_global          %*% Proj_X_to_XY
OutMtx [ 4, ] = stdev_X_global        %*% Proj_X_to_XY
OutMtx [ 5, ] = stdev_X_stratified    %*% Proj_X_to_XY
OutMtx [ 6, ] = r_sqr_X_vs_strata     %*% Proj_X_to_XY
OutMtx [ 7, ] = adj_r_sqr_X_vs_strata %*% Proj_X_to_XY
OutMtx [ 8, ] = p_val_X_vs_strata     %*% Proj_X_to_XY

OutMtx [11, ] = Ycols                 %*% Proj_Y_to_XY
OutMtx [12, ] = cnt_Y_global          %*% Proj_Y_to_XY
OutMtx [13, ] = avg_Y_global          %*% Proj_Y_to_XY
OutMtx [14, ] = stdev_Y_global        %*% Proj_Y_to_XY
OutMtx [15, ] = stdev_Y_stratified    %*% Proj_Y_to_XY
OutMtx [16, ] = r_sqr_Y_vs_strata     %*% Proj_Y_to_XY
OutMtx [17, ] = adj_r_sqr_Y_vs_strata %*% Proj_Y_to_XY
OutMtx [18, ] = p_val_Y_vs_strata     %*% Proj_Y_to_XY

OutMtx [21, ] = cnt_XY_global
OutMtx [22, ] = slope_XY_global
OutMtx [23, ] = stdev_slope_XY_global
OutMtx [24, ] = corr_XY_global
OutMtx [25, ] = stdev_errY_vs_X_global
OutMtx [26, ] = r_sqr_X_vs_Y_global
OutMtx [27, ] = adj_r_sqr_X_vs_Y_global
OutMtx [28, ] = p_val_Y_vs_X_global

OutMtx [31, ] = cnt_XY_with_good_stratumID
OutMtx [32, ] = slope_XY_stratified
OutMtx [33, ] = stdev_slope_XY_stratified
OutMtx [34, ] = corr_XY_stratified
OutMtx [35, ] = stdev_errY_vs_X_stratified
OutMtx [36, ] = r_sqr_X_vs_Y_stratified
OutMtx [37, ] = adj_r_sqr_X_vs_Y_stratified
OutMtx [38, ] = p_val_Y_vs_X_stratified
OutMtx [39, ] = colSums (Cnt_XY_per_stratum >= 2)

OutMtx = t(OutMtx)
writeMM(as(OutMtx, "CsparseMatrix"), file=args[2])

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


args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")

X = readMM(paste(args[1], "X.mtx", sep=""))
y = readMM(paste(args[1], "y.mtx", sep=""))

intercept_status = as.integer(args[2]);
tolerance = as.double(args[3]);
max_iteration = as.double(args[4]);
regularization = as.double(args[5]);

n = nrow (X);
m = ncol (X);
ones_n = matrix (1, n, 1);
zero_cell = matrix (0, 1, 1);

m_ext = m;
if (intercept_status == 1 | intercept_status == 2)  # add the intercept column
{
    X = cbind (X, ones_n);
    m_ext = ncol (X);
}

scale_lambda = matrix (1, m_ext, 1);
if (intercept_status == 1 | intercept_status == 2)
{
    scale_lambda [m_ext, 1] = 0;
}

if (intercept_status == 2) {
    avg_X_cols = t(colSums(X)) / n;
    var_X_cols = (t(colSums (X ^ 2)) - n * (avg_X_cols ^ 2)) / (n - 1);
    is_unsafe = (var_X_cols <= 0);
    scale_X = 1.0 / sqrt (var_X_cols * (1 - is_unsafe) + is_unsafe);
    scale_X [m_ext, 1] = 1;
    shift_X = - avg_X_cols * scale_X;
    shift_X [m_ext, 1] = 0;
} else {
    scale_X = matrix (1, m_ext, 1);
    shift_X = matrix (0, m_ext, 1);
}

lambda = scale_lambda * regularization;
beta_unscaled = matrix (0, m_ext, 1);

if (max_iteration == 0) {
    max_iteration = m_ext;
}
i = 0;
r = - t(X) %*% y;

if (intercept_status == 2) {
    r = scale_X * r + shift_X %*% r [m_ext, ];
}

p = - r;
norm_r2 = sum (r ^ 2);
norm_r2_initial = norm_r2;
norm_r2_target = norm_r2_initial * tolerance ^ 2;

while (i < max_iteration & norm_r2 > norm_r2_target)
{
    if (intercept_status == 2) {
        ssX_p = scale_X * p;
        ssX_p [m_ext, ] = ssX_p [m_ext, ] + t(shift_X) %*% p;
    } else {
        ssX_p = p;
    }
    
    q = t(X) %*% (X %*% ssX_p);

    if (intercept_status == 2) {
        q = scale_X * q + shift_X %*% q [m_ext, ];
    }

	q = q + lambda * p;
	a = norm_r2 / sum (p * q);
	beta_unscaled = beta_unscaled + a * p;
	r = r + a * q;
	old_norm_r2 = norm_r2;
	norm_r2 = sum (r ^ 2);
	p = -r + (norm_r2 / old_norm_r2) * p;
	i = i + 1;
}

if (intercept_status == 2) {
    beta = scale_X * beta_unscaled;
    beta [m_ext, ] = beta [m_ext, ] + t(shift_X) %*% beta_unscaled;
} else {
    beta = beta_unscaled;
}

avg_tot = sum (y) / n;
ss_tot = sum (y ^ 2);
ss_avg_tot = ss_tot - n * avg_tot ^ 2;
var_tot = ss_avg_tot / (n - 1);
y_residual = y - X %*% beta;
avg_res = sum (y_residual) / n;
ss_res = sum (y_residual ^ 2);
ss_avg_res = ss_res - n * avg_res ^ 2;

plain_R2 = 1 - ss_res / ss_avg_tot;
if (n > m_ext) {
    dispersion  = ss_res / (n - m_ext);
    adjusted_R2 = 1 - dispersion / (ss_avg_tot / (n - 1));
} else {
    dispersion  = 0.0 / 0.0;
    adjusted_R2 = 0.0 / 0.0;
}

plain_R2_nobias = 1 - ss_avg_res / ss_avg_tot;
deg_freedom = n - m - 1;
if (deg_freedom > 0) {
    var_res = ss_avg_res / deg_freedom;
    adjusted_R2_nobias = 1 - var_res / (ss_avg_tot / (n - 1));
} else {
    var_res = 0.0 / 0.0;
    adjusted_R2_nobias = 0.0 / 0.0;
}

plain_R2_vs_0 = 1 - ss_res / ss_tot;
if (n > m) {
    adjusted_R2_vs_0 = 1 - (ss_res / (n - m)) / (ss_tot / n);
} else {
    adjusted_R2_vs_0 = 0.0 / 0.0;
}

if (intercept_status == 2) {
    beta_out = cbind (beta, beta_unscaled);
} else {
    beta_out = beta;
}

writeMM(as(beta_out,"CsparseMatrix"), paste(args[6], "w", sep=""))

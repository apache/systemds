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
library("Matrix")
library("matrixStats")

X = readMM(paste(args[1], "X.mtx", sep=""));
Y_vec = readMM(paste(args[1], "Y.mtx", sep=""));
intercept = as.integer(args[2]);
tol = as.double(args[3]);
maxiter = as.integer(args[4]);

intercept_status = intercept;
regularization = 0.001;
maxinneriter = 0;

print ("BEGIN MULTINOMIAL LOGISTIC REGRESSION SCRIPT");

eta0 = 0.0001;
eta1 = 0.25;
eta2 = 0.75;
sigma1 = 0.25;
sigma2 = 0.5;
sigma3 = 4.0;
psi = 0.1;

N = nrow (X);
D = ncol (X);

# Introduce the intercept, shift and rescale the columns of X if needed
if (intercept_status == 1 | intercept_status == 2)  # add the intercept column
{
    X = cbind (X, matrix (1, N, 1));
    D = ncol (X);
}

scale_lambda = matrix (1, D, 1);
if (intercept_status == 1 | intercept_status == 2)
{
    scale_lambda [D, 1] = 0;
}

if (intercept_status == 2)  # scale-&-shift X columns to mean 0, variance 1
{                           # Important assumption: X [, D] = matrix (1, rows = N, cols = 1)
    avg_X_cols = t(colSums(X)) / N;
    var_X_cols = (t(colSums (X ^ 2)) - N * (avg_X_cols ^ 2)) / (N - 1);
    is_unsafe = (var_X_cols <= 0.0);
    scale_X = 1.0 / sqrt (var_X_cols * (1 - is_unsafe) + is_unsafe);
    scale_X [D, 1] = 1;
    shift_X = - avg_X_cols * scale_X;
    shift_X [D, 1] = 0;
    rowSums_X_sq = (X ^ 2) %*% (scale_X ^ 2) + X %*% (2 * scale_X * shift_X) + sum (shift_X ^ 2);
} else {
    scale_X = matrix (1, D, 1);
    shift_X = matrix (0, D, 1);
    rowSums_X_sq = rowSums (X ^ 2);
}

# Henceforth we replace "X" with "X %*% (SHIFT/SCALE TRANSFORM)" and rowSums(X ^ 2)
# with "rowSums_X_sq" in order to preserve the sparsity of X under shift and scale.
# The transform is then associatively applied to the other side of the expression,
# and is rewritten via "scale_X" and "shift_X" as follows:
#
# ssX_A  = (SHIFT/SCALE TRANSFORM) %*% A    --- is rewritten as:
# ssX_A  = diag (scale_X) %*% A;
# ssX_A [D, ] = ssX_A [D, ] + t(shift_X) %*% A;
#
# tssX_A = t(SHIFT/SCALE TRANSFORM) %*% A   --- is rewritten as:
# tssX_A = diag (scale_X) %*% A + shift_X %*% A [D, ];

# Convert "Y_vec" into indicator matrice:
if (min (Y_vec) <= 0) { 
    # Category labels "0", "-1" etc. are converted into the largest label
    max_y = max (Y_vec);
    Y_vec  = Y_vec  + (- Y_vec  + max_y + 1) * (Y_vec <= 0.0);
}
Y = table (seq (1, N, 1), as.vector(Y_vec));
Y = as.matrix(as.data.frame.matrix(Y)) #this is required due to different table semantics

K = ncol (Y) - 1;   # The number of  non-baseline categories

lambda = (scale_lambda %*% matrix (1, 1, K)) * regularization;
delta = 0.5 * sqrt (D) / max (sqrt (rowSums_X_sq));

B = matrix (0, D, K);     ### LT = X %*% (SHIFT/SCALE TRANSFORM) %*% B;
                                        ### LT = append (LT, matrix (0, rows = N, cols = 1));
                                        ### LT = LT - rowMaxs (LT) %*% matrix (1, rows = 1, cols = K+1);
P = matrix (1, N, K+1);   ### exp_LT = exp (LT);
P = P / (K + 1);                        ### P =  exp_LT / (rowSums (exp_LT) %*% matrix (1, rows = 1, cols = K+1));
obj = N * log (K + 1);                  ### obj = - sum (Y * LT) + sum (log (rowSums (exp_LT))) + 0.5 * sum (lambda * (B_new ^ 2));

Grad = t(X) %*% (P [, 1:K] - Y [, 1:K]);
if (intercept_status == 2) {
    Grad = diag (scale_X) %*% Grad + shift_X %*% Grad [D, ];
}
Grad = Grad + lambda * B;
norm_Grad = sqrt (sum (Grad ^ 2));
norm_Grad_initial = norm_Grad;

if (maxinneriter == 0) {
    maxinneriter = D * K;
}
iter = 1;

# boolean for convergence check
converge = (norm_Grad < tol) | (iter > maxiter);

print (paste("-- Initially:  Objective = ", obj, ",  Gradient Norm = ", norm_Grad , ",  Trust Delta = " , delta));

while (! converge)
{
	# SOLVE TRUST REGION SUB-PROBLEM
	S = matrix (0, D, K);
	R = - Grad;
	V = R;
	delta2 = delta ^ 2;
	inneriter = 1;
	norm_R2 = sum (R ^ 2);
	innerconverge = (sqrt (norm_R2) <= psi * norm_Grad);
	is_trust_boundary_reached = 0;

	while (! innerconverge)
	{
	    if (intercept_status == 2) {
	        ssX_V = diag (scale_X) %*% V;
	        ssX_V [D, ] = ssX_V [D, ] + t(shift_X) %*% V;
	    } else {
	        ssX_V = V;
	    }
        Q = P [, 1:K] * (X %*% ssX_V);
        HV = t(X) %*% (Q - P [, 1:K] * (rowSums (Q) %*% matrix (1, 1, K)));
        if (intercept_status == 2) {
            HV = diag (scale_X) %*% HV + shift_X %*% HV [D, ];
        }
        HV = HV + lambda * V;
		alpha = norm_R2 / sum (V * HV);
		Snew = S + alpha * V;
		norm_Snew2 = sum (Snew ^ 2);
		if (norm_Snew2 <= delta2)
		{
			S = Snew;
			R = R - alpha * HV;
			old_norm_R2 = norm_R2 
			norm_R2 = sum (R ^ 2);
			V = R + (norm_R2 / old_norm_R2) * V;
			innerconverge = (sqrt (norm_R2) <= psi * norm_Grad);
		} else {
	        is_trust_boundary_reached = 1;
			sv = sum (S * V);
			v2 = sum (V ^ 2);
			s2 = sum (S ^ 2);
			rad = sqrt (sv ^ 2 + v2 * (delta2 - s2));
			if (sv >= 0) {
				alpha = (delta2 - s2) / (sv + rad);
			} else {
				alpha = (rad - sv) / v2;
			}
			S = S + alpha * V;
			R = R - alpha * HV;
			innerconverge = TRUE;
		}
	    inneriter = inneriter + 1;
	    innerconverge = innerconverge | (inneriter > maxinneriter);
	}  
	
	# END TRUST REGION SUB-PROBLEM
	
	# compute rho, update B, obtain delta
	gs = sum (S * Grad);
	qk = - 0.5 * (gs - sum (S * R));
	B_new = B + S;
	if (intercept_status == 2) {
	    ssX_B_new = diag (scale_X) %*% B_new;
	    ssX_B_new [D, ] = ssX_B_new [D, ] + t(shift_X) %*% B_new;
    } else {
        ssX_B_new = B_new;
    }
    
    LT = as.matrix(cbind ((X %*% ssX_B_new), matrix (0, N, 1)));
    LT = LT - rowMaxs (LT) %*% matrix (1, 1, K+1);
    exp_LT = exp (LT);
    P_new  = exp_LT / (rowSums (exp_LT) %*% matrix (1, 1, K+1));
    obj_new = - sum (Y * LT) + sum (log (rowSums (exp_LT))) + 0.5 * sum (lambda * (B_new ^ 2));
    	
	# Consider updating LT in the inner loop
	# Consider the big "obj" and "obj_new" rounding-off their small difference below:

	actred = (obj - obj_new);
	
	rho = actred / qk;
	is_rho_accepted = (rho > eta0);
	snorm = sqrt (sum (S ^ 2));

	if (iter == 1) {
	   delta = min (delta, snorm);
	}

	alpha2 = obj_new - obj - gs;
	if (alpha2 <= 0) {
	   alpha = sigma3;
	} 
	else {
	   alpha = max (sigma1, -0.5 * gs / alpha2);
	}
	
	if (rho < eta0) {
		delta = min (max (alpha, sigma1) * snorm, sigma2 * delta);
	}
	else {
		if (rho < eta1) {
			delta = max (sigma1 * delta, min (alpha * snorm, sigma2 * delta));
		}
		else { 
			if (rho < eta2) {
				delta = max (sigma1 * delta, min (alpha * snorm, sigma3 * delta));
			}
			else {
				delta = max (delta, min (alpha * snorm, sigma3 * delta));
			}
		}
	} 
	
	if (is_trust_boundary_reached == 1)
	{
	    print (paste("-- Outer Iteration " , iter , ": Had " , (inneriter - 1) , " CG iterations, trust bound REACHED"));
	} else {
	    print (paste("-- Outer Iteration " , iter , ": Had " , (inneriter - 1) , " CG iterations"));
	}
	print (paste("   -- Obj.Reduction:  Actual = " , actred , ",  Predicted = " , qk , 
	       "  (A/P: " , (round (10000.0 * rho) / 10000.0) , "),  Trust Delta = " , delta));
	       
	if (is_rho_accepted)
	{
		B = B_new;
		P = P_new;
		Grad = t(X) %*% (P [, 1:K] - Y [, 1:K]);
		if (intercept_status == 2) {
		    Grad = diag (scale_X) %*% Grad + shift_X %*% Grad [D, ];
		}
		Grad = Grad + lambda * B;
		norm_Grad = sqrt (sum (Grad ^ 2));
		obj = obj_new;
	    print (paste("   -- New Objective = " , obj , ",  Beta Change Norm = " , snorm , ",  Gradient Norm = " , norm_Grad));
	} 
	
	iter = iter + 1;
	converge = ((norm_Grad < (tol * norm_Grad_initial)) | (iter > maxiter) |
	    ((is_trust_boundary_reached == 0) & (abs (actred) < (abs (obj) + abs (obj_new)) * 0.00000000000001)));
    if (converge) { print ("Termination / Convergence condition satisfied."); } else { print (" "); }
} 

if (intercept_status == 2) {
    B_out = diag (scale_X) %*% B;
    B_out [D, ] = B_out [D, ] + t(shift_X) %*% B;
} else {
    B_out = B;
}

writeMM(as(B_out,"CsparseMatrix"), paste(args[5], "w", sep=""));

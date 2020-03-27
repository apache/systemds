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
options(digits = 22)
library("Matrix")

reorder_matrix <- function(
  ncolX, B, S) {
  # This function assumes that B and S have same number of elements.
  # if the intercept is included in the model, all inputs should be adjusted
  # appropriately before calling this function.
  S = t(S);
  B = as.matrix(B);
  num_empty_B = ncolX - nrow(B);

  if (num_empty_B < 0) {
    stop("Error: unable to re-order the matrix. Reason: B more than matrix X");
  }

  if (num_empty_B > 0) {
    pad_zeros = matrix(0, nrow = num_empty_B, ncol = 1);
    B = rbind(B, pad_zeros);
    S = rbind(S, pad_zeros);
  }

  S0 = matrix(0, nrow = nrow(S), ncol = ncol(S));
  S0 = S;

  # since the table won't accept zeros as index we hack it.
  for (a in 1:nrow(S)) {
    for (b in 1:ncol(S)) {

      if (S[a, b] == 0) {

        S0[a, b] = ncolX + 1;
      }

    }
  }
  seqS = seq(1, nrow(S0));

  P = matrix(0, nrow = ncolX, ncol = ncolX);

  for (a in 1:ncolX) {
    for (b in 1:ncolX) {

      if (nrow(S0) < b) {

        P[a, b] = 0;
      } else if (seqS[b] > ncolX | S0[b] > ncolX) {
      } else {

        x = seqS[b];
        y = S0[b];
        P[x, y] = 1;
      }
    }
  }

  Y = t(P) %*% B;
  return(Y)
}

thr = 0.001;
intercept_status = 1;
X = as.matrix(readMM(paste(args[1], "A.mtx", sep = "")))
y = as.matrix(readMM(paste(args[1], "B.mtx", sep = "")))

# currently only the forward selection strategy in supported: start from one feature and iteratively add
# features until AIC improves
dir = 0;
stop = 0;
intercept_status = 1;
thr = 0.001;

print("BEGIN STEPWISE LINEAR REGRESSION SCRIPT");
print("Reading X and Y...");
X_orig = X;

n = nrow(X_orig);
m_orig = ncol(X_orig);

# BEGIN STEPWISE LINEAR REGRESSION

if (dir == 0) {
  continue = TRUE;
  columns_fixed = matrix(0, 1, m_orig);
  columns_fixed_ordered = matrix(0, 1, 1);

  # X_global stores the best model found at each step
  X_global = matrix(0, n, 1);

  if (intercept_status == 1 | intercept_status == 2) {
    beta = mean(y);
    AIC_best = 2 + n * log(sum((beta - y) ^ 2) / n);
  } else {
    beta = 0.0;
    AIC_best = n * log(sum(y ^ 2) / n);
  }

  AICs = matrix(t(AIC_best), 1, m_orig);
  
  boa_ncol = ncol(X_orig);
  if (intercept_status != 0) {
    boa_ncol = boa_ncol + 1
  }

  beta_out_all = matrix(0, boa_ncol, m_orig * 1);
  y_ncol = 1;
  column_best = 0;
  
  # First pass to examine single features
  for (i in 1:m_orig) {
    columns_fixed_ordered_1 = as.matrix(i);
    x = as.matrix(X_orig[, i]);
		beta_out_i = as.matrix(lm.fit(x, y)$coefficients)
    
    # COMPUTE AIC
    y_residual = y - x %*% beta_out_i;
    ss_res = sum(y_residual ^ 2);
    eq_deg_of_freedom = ncol(x);
    AIC_1 = (2 * eq_deg_of_freedom) + n * log(ss_res / n);

    AICs[1, i] = AIC_1;
    AIC_cur = AICs[1, i];
    if ((AIC_cur < AIC_best) & ((AIC_best - AIC_cur) > abs(thr * AIC_best))) {
      column_best = i;
      AIC_best = AIC_cur;
    }
    beta_out_all[1:nrow(beta_out_i), ((i - 1) * y_ncol + 1):(i * y_ncol)] = beta_out_i[,1];
  }

  # beta best so far
  beta_best = beta_out_all[, ((column_best - 1) * y_ncol + 1):(column_best * y_ncol)];

  if (column_best == 0) {

    Selected = matrix(0, nrow = 1, ncol = 1);
    if (intercept_status == 0) {
      B = matrix(beta, nrow = m_orig, ncol = 1);
    } else {
      B_tmp = matrix(0, nrow = m_orig + 1, ncol = 1);
      B_tmp[m_orig + 1,] = beta;
      B = B_tmp;
    }

    beta_out = B;
    writeMM(as(beta_out, "CsparseMatrix"), paste(args[2], "C", sep = ""));
    writeMM(as(Selected, "CsparseMatrix"), paste(args[2], "S", sep = ""));

    stop = 1;
  }

  columns_fixed[1, column_best] = 1;
  columns_fixed_ordered[1, 1] = column_best;
  X_global = X_orig[, column_best];

  while (continue) {
    # Subsequent passes over the features

    beta_out_all_2 = matrix(0, boa_ncol, m_orig * 1);

    for (i in 1:m_orig) {
      if (columns_fixed[1, i] == 0) {

        # Construct the feature matrix
        X = cbind(X_global, X_orig[, i]);

        tmp = matrix(0, nrow = 1, ncol = 1);
        tmp[1, 1] = i;
        columns_fixed_ordered_2 = append(columns_fixed_ordered, tmp);


        x = as.matrix(X);
        n = nrow(x);
        m = ncol(x);

        # Introduce the intercept, shift and rescale the columns of X if needed
        if (intercept_status == 1 | intercept_status == 2) {
          # add the intercept column
          ones_n = matrix(1, nrow = n, ncol = 1);
          x = cbind(X_orig[, i], ones_n);
          m = m - 1;
        }

        m_ext = ncol(x);

        # BEGIN THE DIRECT SOLVE ALGORITHM (EXTERNAL CALL)
        beta_out_i = lm.fit(x, y)$coefficients

        # COMPUTE AIC
        y_residual = y - x %*% beta_out_i;
        ss_res = sum(y_residual ^ 2);
        eq_deg_of_freedom = m_ext;
        AIC_2 = (2 * eq_deg_of_freedom) + n * log(ss_res / n);

        b = as.matrix(beta_out_i)
        beta_out_all_2[1:nrow(b), i:i] = b[1, 1];

        if ((AIC_2 < AIC_best) & ((AIC_best - AIC_2) > abs(thr * AIC_best)) & (columns_fixed[1, i] == 0)) {
          column_best = i;
          AIC_best = AIC_2;
        }
      }
    }

    # have the best beta store in the matrix
    beta_best = beta_out_all_2[, ((column_best - 1) * y_ncol + 1):(column_best * y_ncol)];

    # Append best found features (i.e., columns) to X_global
    if (is.null(columns_fixed[1, column_best])) {
      # new best feature found
      columns_fixed[1, column_best] = 1;
      columns_fixed_ordered = cbind(columns_fixed_ordered, as.matrix(column_best));

      if (ncol(columns_fixed_ordered) == m_orig) {
        # all features examined
        X_global = cbind(X_global, X_orig[, column_best]);
        continue = FALSE;
      } else {
        X_global = cbind(X_global, X_orig[, column_best]);
      }
    } else {
      continue = FALSE;
    }
  }

  if (stop == 0) {

    # run linear regression with selected set of features
    print("Running linear regression with selected features...");

    x = as.matrix(X_global);
    n = nrow(x);
    m = ncol(x);

    # Introduce the intercept, shift and rescale the columns of X if needed
    if (intercept_status == 1 | intercept_status == 2) {
      # add the intercept column
      ones_n = matrix(1, nrow = n, ncol = 1);
      x = cbind(X_orig[, i], ones_n);
      m = m - 1;
    }

    m_ext = ncol(x);

    # BEGIN THE DIRECT SOLVE ALGORITHM (EXTERNAL CALL)
    beta_out = lm.fit(x, y)$coefficients

    # COMPUTE AIC
    y_residual = y - x %*% beta_out;
    ss_res = sum(y_residual ^ 2);
    eq_deg_of_freedom = m_ext;
    AIC = (2 * eq_deg_of_freedom) + n * log(ss_res / n);

    Selected = columns_fixed_ordered;
    if (intercept_status != 0) {
      Selected = cbind(Selected, matrix(boa_ncol, nrow = 1, ncol = 1))
    }

    beta_out = reorder_matrix(boa_ncol, beta_out, Selected);

    writeMM(as(beta_out, "CsparseMatrix"), paste(args[2], "C", sep = ""));
    writeMM(as(Selected[1], "CsparseMatrix"), paste(args[2], "S", sep = ""));
  }

} else {
  stop("Currently only forward selection strategy is supported!");
}

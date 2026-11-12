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

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3L) {
  stop("Expected arguments: <mode> <output_path> <data>")
}

mode <- args[1]
output_path <- args[2]

ts_data <- scan(
  text = args[3],
  what = double(),
  sep = ",",
  quiet = TRUE
)

if (length(ts_data) == 0L || any(!is.finite(ts_data))) {
  stop("Input data must contain finite numeric values")
}


if (mode == "double") {
  hw_model <- stats::HoltWinters(
    ts_data,
    alpha=0.3,
    beta = 0.1,
    gamma = FALSE)
  fit_vals <- as.numeric(fitted(hw_model)[, "xhat"])

  res_matrix <- matrix(0, nrow = length(fit_vals), ncol = 1)
  res_matrix[, 1] <- fit_vals

} else if (mode == "triple") {
   ts_obj <- ts(ts_data, frequency = 12)

   hw_model <- stats::HoltWinters(
     ts_obj,
     alpha = 0.3,
     beta = 0.1,
     gamma = 0.2,
     seasonal = "additive"
   )

   fit_vals <- as.numeric(fitted(hw_model)[, "xhat"])

   res_matrix <- matrix(0, nrow = length(fit_vals), ncol = 1)
   res_matrix[, 1] <- fit_vals

} else if (mode == "odd_triple") {
     ts_obj <- ts(ts_data, frequency = 7)

     hw_model <- stats::HoltWinters(
          ts_obj,
          alpha = 0.3,
          beta = 0.1,
          gamma = 0.2,
          seasonal = "additive"
        )

     fit_vals <- as.numeric(fitted(hw_model)[, "xhat"])

     res_matrix <- matrix(0, nrow = length(fit_vals), ncol = 1)
     res_matrix[, 1] <- fit_vals

  } else {
  stop(paste("Unknown smoothing mode:", mode))
}

dir.create(
  dirname(output_path),
  recursive = TRUE,
  showWarnings = FALSE
)

Matrix::writeMM(
  Matrix::Matrix(res_matrix, sparse = TRUE),
  file = output_path
)

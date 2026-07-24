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

# Independently applies a SystemDS Isolation Forest model in R. The model is
# trained by DML and serialized as MatrixMarket, so this script validates tree
# traversal and anomaly-score calculation without relying on matching random
# number generators between SystemDS and R.
#
# Arguments:
#   1. Linearized Isolation Forest model in MatrixMarket format
#   2. Samples to score in MatrixMarket format
#   3. Effective training subsampling size
#   4. Output anomaly scores in MatrixMarket format
#   5. Output Apply runtime in seconds as a 1x1 MatrixMarket matrix

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5)
  stop("Expected model, samples, subsampling size, score output, and runtime output arguments.")

suppressPackageStartupMessages(library("Matrix"))

model <- as.matrix(readMM(args[1]))
X <- as.matrix(readMM(args[2]))
subsampling_size <- as.integer(args[3])

average_path_length <- function(n) {
  if (n <= 1)
    stop("average_path_length requires n > 1")

  if (n < 1000)
    harmonic <- sum(1 / seq_len(n - 1))
  else
    harmonic <- log(n - 1) + 0.57721566490153

  2 * harmonic - 2 * (n - 1) / n
}

tree_path_length <- function(tree, x) {
  node_id <- 1L
  edges <- 0L

  repeat {
    node_start <- 2L * node_id - 1L
    if (node_start + 1L > length(tree))
      stop("Invalid iTree model: node index is out of bounds.")

    split_feature <- as.integer(round(tree[node_start]))
    node_value <- tree[node_start + 1L]

    if (split_feature > 0L) {
      if (split_feature > length(x))
        stop("Invalid iTree model: split feature exceeds the input width.")

      edges <- edges + 1L
      if (x[split_feature] < node_value)
        node_id <- 2L * node_id
      else
        node_id <- 2L * node_id + 1L
    }
    else if (split_feature == 0L) {
      leaf_size <- as.integer(round(node_value))
      if (leaf_size < 1L)
        stop("Invalid iTree model: external-node size must be positive.")

      if (leaf_size <= 1L)
        return(as.numeric(edges))
      return(edges + average_path_length(leaf_size))
    }
    else {
      stop("Invalid iTree model: reached a placeholder node.")
    }
  }
}

score_forest <- function(model, X, subsampling_size) {
  if (nrow(model) < 1L)
    stop("The model must contain at least one tree.")
  if (nrow(X) < 1L)
    stop("X must contain at least one row.")
  if (subsampling_size <= 1L)
    stop("subsampling_size must be greater than one.")

  height_limit <- ceiling(log(subsampling_size, base = 2))
  expected_columns <- 2 * (2^(height_limit + 1) - 1)
  if (ncol(model) != expected_columns)
    stop("The model has an invalid number of columns.")

  normalization <- average_path_length(subsampling_size)
  num_samples <- nrow(X)
  scores <- matrix(0, nrow = num_samples, ncol = 1L)

  for (sample_id in seq_len(num_samples)) {
    path_sum <- 0
    for (tree_id in seq_len(nrow(model)))
      path_sum <- path_sum + tree_path_length(model[tree_id, ], X[sample_id, ])

    scores[sample_id, 1L] <- 2^(-(path_sum / nrow(model)) / normalization)
  }

  scores
}

timing <- system.time({
  scores <- score_forest(model, X, subsampling_size)
})
apply_runtime <- unname(timing[["elapsed"]])

invisible(writeMM(Matrix(scores, sparse = TRUE), args[4]))
invisible(writeMM(Matrix(matrix(apply_runtime, nrow = 1L), sparse = TRUE), args[5]))

cat(sprintf("R Isolation Forest Apply runtime: %.6f s\n", apply_runtime))

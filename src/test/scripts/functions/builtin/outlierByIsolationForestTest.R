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
#   6. Largest benchmark feature matrix in MatrixMarket format
#   7. SystemDS training measurements in MatrixMarket format
#   8. Smallest benchmark row count
#   9. Number of successively doubled benchmark sizes
#  10. Repetitions per benchmark size
#  11. Trees per benchmark forest
#  12. Benchmark subsampling size
#  13. Benchmark seed (repetition is added to it)
#  14. Output R training measurements in MatrixMarket format
#  15. Output median-runtime CSV table
#  16. Output runtime-versus-data-size PNG plot
#  17. Output speedup CSV table

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 17)
  stop("Expected 17 apply-validation and training-benchmark arguments.")

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

# This trainer follows the Isolation Forest algorithm and model layout, but
# intentionally uses R's RNG, sample.int, and runif. Consequently, an equal
# numeric seed makes each implementation reproducible in isolation; it does
# not define a common cross-runtime random stream or an identical forest.
train_tree <- function(X, max_depth) {
  tree_size <- 2L * (2L^(max_depth + 1L) - 1L)
  tree <- rep(-1, tree_size)

  grow <- function(X_node, node_id, depth) {
    node_start <- 2L * node_id - 1L
    ranges <- apply(X_node, 2L, max) - apply(X_node, 2L, min)

    if (nrow(X_node) <= 1L || depth >= max_depth || !any(ranges > 0)) {
      tree[node_start] <<- 0
      tree[node_start + 1L] <<- nrow(X_node)
      return(invisible(NULL))
    }

    valid_features <- which(ranges > 0)
    split_feature <- valid_features[sample.int(length(valid_features), 1L)]
    feature_min <- min(X_node[, split_feature])
    feature_max <- max(X_node[, split_feature])
    split_value <- runif(1L, min = feature_min, max = feature_max)
    goes_left <- X_node[, split_feature] < split_value

    # This matches the defensive leaf fallback in the DML implementation.
    if (!any(goes_left) || all(goes_left)) {
      tree[node_start] <<- 0
      tree[node_start + 1L] <<- nrow(X_node)
      return(invisible(NULL))
    }

    tree[node_start] <<- split_feature
    tree[node_start + 1L] <<- split_value
    grow(X_node[goes_left, , drop = FALSE], 2L * node_id, depth + 1L)
    grow(X_node[!goes_left, , drop = FALSE], 2L * node_id + 1L, depth + 1L)
    invisible(NULL)
  }

  grow(X, 1L, 0L)
  tree
}

train_forest <- function(X, n_trees, requested_subsampling_size, seed) {
  if (nrow(X) <= 1L)
    stop("Training data must contain at least two rows.")
  if (n_trees <= 0L)
    stop("n_trees must be positive.")

  effective_size <- min(requested_subsampling_size, nrow(X))
  if (effective_size <= 1L)
    stop("subsampling_size must be greater than one.")

  if (seed >= 0L)
    set.seed(seed)

  height_limit <- ceiling(log(effective_size, base = 2))
  tree_size <- 2L * (2L^(height_limit + 1L) - 1L)
  forest <- matrix(-1, nrow = n_trees, ncol = tree_size)

  for (tree_id in seq_len(n_trees)) {
    sampled_rows <- sample.int(nrow(X), effective_size, replace = FALSE)
    forest[tree_id, ] <- train_tree(X[sampled_rows, , drop = FALSE], height_limit)
  }

  forest
}

run_training_benchmark <- function(X, base_rows, num_sizes, repetitions,
    n_trees, benchmark_subsampling_size, benchmark_seed) {
  sizes <- as.integer(base_rows * 2^(seq_len(num_sizes) - 1L))
  if (max(sizes) > nrow(X))
    stop("The benchmark feature matrix is smaller than the largest requested size.")

  measurements <- matrix(0, nrow = num_sizes * repetitions, ncol = 5L)
  measurement_id <- 1L

  for (data_size in sizes) {
    X_size <- X[seq_len(data_size), , drop = FALSE]
    for (repetition in seq_len(repetitions)) {
      repetition_seed <- as.integer(benchmark_seed + repetition)
      timing <- system.time({
        forest <- train_forest(
          X_size, n_trees, benchmark_subsampling_size, repetition_seed)
        model_checksum <- sum(forest)
        model_squared_checksum <- sum(forest * forest)
      })

      # elapsed can be reported as zero on coarse timers; use the smallest
      # positive double so logarithmic plots and speedup ratios remain defined.
      elapsed <- max(unname(timing[["elapsed"]]), .Machine$double.eps)
      measurements[measurement_id, ] <- c(
        data_size,
        repetition,
        elapsed,
        model_checksum,
        model_squared_checksum
      )
      measurement_id <- measurement_id + 1L
    }
  }

  measurements
}

timing <- system.time({
  scores <- score_forest(model, X, subsampling_size)
})
apply_runtime <- unname(timing[["elapsed"]])

invisible(writeMM(Matrix(scores, sparse = TRUE), args[4]))
invisible(writeMM(Matrix(matrix(apply_runtime, nrow = 1L), sparse = TRUE), args[5]))

cat(sprintf("R Isolation Forest Apply runtime: %.6f s\n", apply_runtime))

benchmark_X <- as.matrix(readMM(args[6]))
dml_training_runtimes <- as.matrix(readMM(args[7]))
benchmark_base_rows <- as.integer(args[8])
benchmark_num_sizes <- as.integer(args[9])
benchmark_repetitions <- as.integer(args[10])
benchmark_n_trees <- as.integer(args[11])
benchmark_subsampling_size <- as.integer(args[12])
benchmark_seed <- as.integer(args[13])

r_training_runtimes <- run_training_benchmark(
  benchmark_X,
  benchmark_base_rows,
  benchmark_num_sizes,
  benchmark_repetitions,
  benchmark_n_trees,
  benchmark_subsampling_size,
  benchmark_seed
)

expected_measurements <- benchmark_num_sizes * benchmark_repetitions
if (nrow(dml_training_runtimes) != expected_measurements ||
    ncol(dml_training_runtimes) != 5L)
  stop("SystemDS benchmark output has unexpected dimensions.")
if (!all(dml_training_runtimes[, 1:2, drop = FALSE] ==
    r_training_runtimes[, 1:2, drop = FALSE]))
  stop("SystemDS and R benchmark measurements are not aligned by size and repetition.")

# The same data and numeric seed are deliberately used in corresponding rows.
# At least one of two fingerprints must differ, demonstrating that the seed is
# not a portable forest specification across the two RNG implementations.
checksum_equal <- abs(dml_training_runtimes[, 4] - r_training_runtimes[, 4]) <=
  1e-12 * pmax(1, abs(dml_training_runtimes[, 4]), abs(r_training_runtimes[, 4]))
squared_checksum_equal <- abs(dml_training_runtimes[, 5] - r_training_runtimes[, 5]) <=
  1e-12 * pmax(1, abs(dml_training_runtimes[, 5]), abs(r_training_runtimes[, 5]))
if (all(checksum_equal & squared_checksum_equal))
  stop("R and SystemDS unexpectedly produced identical forest fingerprints.")

benchmark_sizes <- as.integer(benchmark_base_rows *
  2^(seq_len(benchmark_num_sizes) - 1L))
systemds_medians <- vapply(benchmark_sizes, function(data_size) {
  median(dml_training_runtimes[dml_training_runtimes[, 1] == data_size, 3])
}, numeric(1L))
r_medians <- vapply(benchmark_sizes, function(data_size) {
  median(r_training_runtimes[r_training_runtimes[, 1] == data_size, 3])
}, numeric(1L))

median_table <- data.frame(
  data_size = benchmark_sizes,
  systemds_median_seconds = systemds_medians,
  r_median_seconds = r_medians,
  systemds_speedup_vs_r = r_medians / systemds_medians
)
speedup_table <- median_table[, c("data_size", "systemds_speedup_vs_r")]

invisible(writeMM(Matrix(r_training_runtimes, sparse = TRUE), args[14]))
write.csv(median_table, args[15], row.names = FALSE, quote = FALSE)
write.csv(speedup_table, args[17], row.names = FALSE, quote = FALSE)

png(args[16], width = 960L, height = 640L)
matplot(
  benchmark_sizes,
  cbind(systemds_medians, r_medians),
  type = "o",
  log = "xy",
  lty = 1L,
  lwd = 2L,
  pch = c(16L, 17L),
  col = c("#1f77b4", "#d62728"),
  xaxt = "n",
  xlab = "Training rows (log2 scale)",
  ylab = "Median training runtime (seconds, log scale)",
  main = sprintf("Isolation Forest training runtime (%d repetitions)", benchmark_repetitions)
)
axis(1L, at = benchmark_sizes, labels = benchmark_sizes)
grid()
legend(
  "topleft",
  legend = c("SystemDS", "R"),
  col = c("#1f77b4", "#d62728"),
  lty = 1L,
  lwd = 2L,
  pch = c(16L, 17L),
  bty = "n"
)
invisible(dev.off())

cat("Isolation Forest training median-runtime table:\n")
print(median_table, row.names = FALSE)
cat(paste0(
  "Small-data note: fixed SystemDS costs (JVM/DML startup and compilation in end-to-end runs, ",
  "plus per-training parfor scheduling and task setup) can dominate useful tree-building work. ",
  "The in-script timings in this table exclude the one-off JVM/script startup but retain the ",
  "per-call execution setup, so small sizes should not be interpreted as pure row-scaling costs.\n"
))
cat(paste0(
  "Seed note: the same numeric seed still produces different independently trained forests in R ",
  "and SystemDS because their RNGs, sampling procedures, and random-number consumption order differ. ",
  "Exact anomaly-score agreement above is tested by applying the same serialized SystemDS forest.\n"
))

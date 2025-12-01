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


# Generate various HDF5 test files with different formats.
# Creates test files in the 'in' directory.

if (!require("rhdf5", quietly = TRUE)) {
  cat("Error: rhdf5 is not installed.\n")
  quit(status = 1)
}

SMALL_MATRIX_2D <- c(200, 40)
SMALL_MATRIX_3D <- c(15, 15, 5)
SMALL_TENSOR_4D_A <- c(120, 16, 16, 4)
SMALL_TENSOR_4D_B <- c(120, 16, 16, 5)
SMALL_LABEL_MATRIX <- c(120, 12)

VECTOR_LENGTH <- 200
STRING_ARRAY_LENGTH <- 30

CHUNK_SHAPE <- c(100, 20)

write_matrix <- function(file_path, dataset_name, shape, generator = function(n) rnorm(n)) {
  values <- generator(prod(shape))
  # Create dataset without compression, filters, or chunking to avoid message type 11 (Filter Pipeline)
  # filter = "NONE": explicitly disable compression filters
  # level = 0: no compression
  # shuffle = FALSE: no shuffle filter
  # chunk = dims: single chunk matching dataset size (effectively contiguous for small datasets)
  h5createDataset(file_path, dataset_name, dims = shape, 
                  filter = "NONE", level = 0, shuffle = FALSE, chunk = shape)
  h5write(array(values, dim = shape), file_path, dataset_name)
}

generate_test_file_single_dataset <- function(dir) {
  file_path <- file.path(dir, "test_single_dataset.h5")
  h5createFile(file_path)
  write_matrix(file_path, "data", SMALL_MATRIX_2D)
  cat("Created test_single_dataset.h5 (single 2D dataset)\n")
}

generate_test_file_multiple_datasets <- function(dir) {
  file_path <- file.path(dir, "test_multiple_datasets.h5")
  h5createFile(file_path)
  write_matrix(file_path, "matrix_2d", SMALL_MATRIX_2D)
  # Create 1D vector without compression/filters
  h5createDataset(file_path, "vector_1d", dims = VECTOR_LENGTH, 
                  filter = "NONE", level = 0, shuffle = FALSE, chunk = VECTOR_LENGTH)
  h5write(rnorm(VECTOR_LENGTH), file_path, "vector_1d")
  write_matrix(file_path, "matrix_3d", SMALL_MATRIX_3D)
  cat("Created test_multiple_datasets.h5 (1D/2D/3D datasets)\n")
}

generate_test_file_different_dtypes <- function(dir) {
  file_path <- file.path(dir, "test_different_dtypes.h5")
  h5createFile(file_path)
  write_matrix(file_path, "double_primary", SMALL_MATRIX_2D)
  write_matrix(file_path, "double_secondary", SMALL_MATRIX_2D)
  write_matrix(
    file_path,
    "int32",
    SMALL_MATRIX_2D,
    generator = function(n) as.integer(sample(-100:100, n, replace = TRUE))
  )
  write_matrix(
    file_path,
    "int32_alt",
    SMALL_MATRIX_2D,
    generator = function(n) as.integer(sample(-100:100, n, replace = TRUE))
  )
  cat("Created test_different_dtypes.h5 (double/int datasets)\n")
}

# https://support.hdfgroup.org/documentation/hdf5-docs/advanced_topics/chunking_in_hdf5.html
generate_test_file_chunked <- function(dir) {
  file_path <- file.path(dir, "test_chunked.h5")
  h5createFile(file_path)

  data <- array(rnorm(prod(SMALL_MATRIX_2D)), dim = SMALL_MATRIX_2D)
  # Chunked dataset without compression/filters (chunking is intentional for this test)
  h5createDataset(file_path, "chunked_data", dims = SMALL_MATRIX_2D, chunk = CHUNK_SHAPE,
                  filter = "NONE", level = 0, shuffle = FALSE)
  h5write(data, file_path, "chunked_data")

  write_matrix(file_path, "non_chunked_data", SMALL_MATRIX_2D)
  cat("Created test_chunked.h5 (chunked dataset)\n")
}

generate_test_file_compressed <- function(dir) {
  file_path <- file.path(dir, "test_compressed.h5")
  h5createFile(file_path)
  data <- array(rnorm(prod(SMALL_MATRIX_2D)), dim = SMALL_MATRIX_2D)
  h5createDataset(file_path, "gzip_compressed_9", dims = SMALL_MATRIX_2D, 
                  chunk = SMALL_MATRIX_2D, level = 9)
  h5write(data, file_path, "gzip_compressed_9")
  h5createDataset(file_path, "gzip_compressed_1", dims = SMALL_MATRIX_2D, 
                  chunk = SMALL_MATRIX_2D, level = 1)
  h5write(data, file_path, "gzip_compressed_1")
  cat("Created test_compressed.h5 (gzip compression)\n")
}

generate_test_file_multi_tensor_samples <- function(dir) {
  file_path <- file.path(dir, "test_multi_tensor_samples.h5")
  h5createFile(file_path)
  write_matrix(
    file_path,
    "sen1",
    SMALL_TENSOR_4D_A
  )
  write_matrix(
    file_path,
    "sen2",
    SMALL_TENSOR_4D_B
  )
  write_matrix(
    file_path,
    "label",
    SMALL_LABEL_MATRIX,
    generator = function(n) as.integer(sample(0:1, n, replace = TRUE))
  )
  cat("Created test_multi_tensor_samples.h5 (multi-input tensors)\n")
}

generate_test_file_nested_groups <- function(dir) {
  file_path <- file.path(dir, "test_nested_groups.h5")
  h5createFile(file_path)
  write_matrix(file_path, "root_data", SMALL_MATRIX_2D)
  h5createGroup(file_path, "group1")
  write_matrix(file_path, "group1/data1", SMALL_MATRIX_2D)
  h5createGroup(file_path, "group1/subgroup")
  write_matrix(file_path, "group1/subgroup/data2", SMALL_MATRIX_2D)
  cat("Created test_nested_groups.h5 (nested group hierarchy)\n")
}

generate_test_file_with_attributes <- function(dir) {
  file_path <- file.path(dir, "test_with_attributes.h5")
  h5createFile(file_path)
  write_matrix(file_path, "data", SMALL_MATRIX_2D)

  fid <- H5Fopen(file_path)
  did <- H5Dopen(fid, "data")
  h5writeAttribute("Test dataset with attributes", did, "description")
  h5writeAttribute(1.0, did, "version")
  h5writeAttribute(SMALL_MATRIX_2D, did, "shape")
  H5Dclose(did)

  h5writeAttribute("2025-11-26", fid, "file_created")
  h5writeAttribute("attributes", fid, "test_type")
  H5Fclose(fid)
  cat("Created test_with_attributes.h5 (dataset + file attributes)\n")
}

generate_test_file_empty_datasets <- function(dir) {
  file_path <- file.path(dir, "test_empty_datasets.h5")
  h5createFile(file_path)
  h5createDataset(file_path, "empty", dims = c(0, SMALL_MATRIX_2D[2]), 
                  filter = "NONE", level = 0, shuffle = FALSE)

  # Create scalar and vector without compression/filters
  h5createDataset(file_path, "scalar", dims = 1, 
                  filter = "NONE", level = 0, shuffle = FALSE, chunk = 1)
  h5write(1.0, file_path, "scalar")
  h5createDataset(file_path, "vector", dims = VECTOR_LENGTH, 
                  filter = "NONE", level = 0, shuffle = FALSE, chunk = VECTOR_LENGTH)
  h5write(rnorm(VECTOR_LENGTH), file_path, "vector")
  cat("Created test_empty_datasets.h5 (empty/scalar/vector)\n")
}

generate_test_file_string_datasets <- function(dir) {
  file_path <- file.path(dir, "test_string_datasets.h5")
  h5createFile(file_path)
  strings <- paste0("string_", 0:(STRING_ARRAY_LENGTH - 1))
  # Create string dataset without compression/filters
  h5createDataset(file_path, "string_array", dims = STRING_ARRAY_LENGTH, 
                  storage.mode = "character", filter = "NONE", level = 0, 
                  shuffle = FALSE, chunk = STRING_ARRAY_LENGTH)
  h5write(strings, file_path, "string_array")
  cat("Created test_string_datasets.h5 (string datasets)\n")
}

main <- function() {
  # Check if working directory is "hdf5". Quit if not.
  if (basename(getwd()) != "hdf5") {
    cat("You must execute this script from the 'hdf5' directory!\n")
    quit(status = 1)
  }
  
  testdir <- "in"
  if (!dir.exists(testdir)) {
    dir.create(testdir)
  }
  
  test_functions <- list(
    generate_test_file_single_dataset,
    generate_test_file_multiple_datasets,
    generate_test_file_different_dtypes,
    generate_test_file_chunked,
    generate_test_file_compressed,
    generate_test_file_multi_tensor_samples,
    generate_test_file_nested_groups,
    generate_test_file_with_attributes,
    generate_test_file_empty_datasets,
    generate_test_file_string_datasets
  )
  
  for (test_func in test_functions) {
    tryCatch({
      test_func(testdir)
    }, error = function(e) {
      cat(sprintf("  ✗ Error: %s\n", conditionMessage(e)))
    })
  }
  
  files <- sort(list.files(testdir, pattern = "\\.h5$", full.names = TRUE))
  cat(sprintf("\nGenerated %d HDF5 test files in %s\n", length(files), normalizePath(testdir)))
}

if (!interactive()) {
  main()
}
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

"""
Preprocess -- Predicting Breast Cancer Proliferation Scores with
Apache SystemML

This script runs the preprocessing phase of the breast cancer project.
"""
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession

from breastcancer.preprocessing import add_row_indices, get_labels_df, preprocess, save


# Create new SparkSession
spark = (SparkSession.builder
                     .appName("Breast Cancer -- Preprocessing")
                     .getOrCreate())

# Ship a fresh copy of the `breastcancer` package to the Spark workers.
# Note: The zip must include the `breastcancer` directory itself,
# as well as all files within it for `addPyFile` to work correctly.
# This is equivalent to `zip -r breastcancer.zip breastcancer`.
dirname = "breastcancer"
zipname = dirname + ".zip"
shutil.make_archive(dirname, 'zip', dirname + "/..", dirname)
spark.sparkContext.addPyFile(zipname)


# Execute Preprocessing & Save

# TODO: Filtering tiles and then cutting into samples could result
# in samples with less tissue than desired, despite that being the
# procedure of the paper.  Look into simply selecting tiles of the
# desired size to begin with.

# Settings
# TODO: Convert this to a set of parsed command line arguments
tile_size = 256
sample_size = 256
grayscale = False
num_partitions = 20000
training = True
row_indices = False
train_frac = 0.8
sample_frac=0.01
seed = 42
folder = "data"  # Linux-filesystem directory to read raw WSI data
save_folder = "data"  # Hadoop-supported directory in which to save DataFrames
train_df_path = os.path.join(save_folder, "train_{}{}.parquet".format(sample_size,
    "_grayscale" if grayscale else ""))
val_df_path = os.path.join(save_folder, "val_{}{}.parquet".format(sample_size,
    "_grayscale" if grayscale else ""))
train_sample_path = os.path.join(save_folder, "train_{}_sample_{}{}.parquet".format(sample_frac,
    sample_size, "_grayscale" if grayscale else ""))
val_sample_path = os.path.join(save_folder, "val_{}_sample_{}{}.parquet".format(sample_frac,
    sample_size, "_grayscale" if grayscale else ""))

# Get labels
labels_df = get_labels_df(folder)

# Split into train and validation sets based on slide number, stratified by class
train, val = train_test_split(labels_df, train_size=train_frac, stratify=labels_df['tumor_score'],
                              random_state=seed)

# Process train & val slides
train_df = preprocess(spark, train.index, tile_size=tile_size, sample_size=sample_size,
                      grayscale=grayscale, num_partitions=num_partitions, folder=folder)
val_df = preprocess(spark, val.index, tile_size=tile_size, sample_size=sample_size,
                    grayscale=grayscale, num_partitions=num_partitions, folder=folder)

if row_indices:
  # Add row indices
  train_df = add_row_indices(train_df)
  val_df = add_row_indices(val_df)

# Save train & val DataFrames
save(train_df, train_df_path, sample_size, grayscale)
save(val_df, val_df_path, sample_size, grayscale)

if sample_frac > 0:
  # Sample Data
  train_df = spark.read.load(train_df_path)
  val_df = spark.read.load(val_df_path)
  train_sample = sample(train_df, sample_frac, seed)
  val_sample = sample(val_df, sample_frac, seed)

  # Save sampled DataFrames.
  save(train_sample, train_sample_path, sample_size, grayscale)
  save(val_sample, val_sample_path, sample_size, grayscale)


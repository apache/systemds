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

from breastcancer.preprocessing import preprocess, save, train_val_split
from pyspark.sql import SparkSession


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

# Get list of image numbers, minus the broken ones.
broken = {2, 45, 91, 112, 242, 256, 280, 313, 329, 467}
slide_nums = sorted(set(range(1,501)) - broken)

# Settings
training = True
tile_size = 256
sample_size = 256
grayscale = False
num_partitions = 20000
add_row_indices = True
train_frac = 0.8
split_seed = 24
folder = "/home/MDM/breast_cancer/data"
save_folder = "data"  # Hadoop-supported directory in which to save DataFrames
df_path = os.path.join(save_folder, "samples_{}_{}{}.parquet".format(
    "labels" if training else "testing", sample_size, "_grayscale" if grayscale else ""))
train_df_path = os.path.join(save_folder, "train_{}{}.parquet".format(sample_size,
    "_grayscale" if grayscale else ""))
val_df_path = os.path.join(save_folder, "val_{}{}.parquet".format(sample_size,
    "_grayscale" if grayscale else ""))

# Process all slides.
df = preprocess(spark, slide_nums, tile_size=tile_size, sample_size=sample_size,
                grayscale=grayscale, training=training, num_partitions=num_partitions,
                folder=folder)

# Save DataFrame of samples.
save(df, df_path, sample_size, grayscale)

# Load full DataFrame from disk.
df = spark.read.load(df_path)

# Split into train and validation DataFrames based On slide number
train, val = train_val_split(spark, df, slide_nums, folder, train_frac, add_row_indices,
                             seed=split_seed)

# Save train and validation DataFrames.
save(train, train_df_path, sample_size, grayscale)
save(val, val_df_path, sample_size, grayscale)


# ---
#
# Sample Data
## TODO: Wrap this in a function with appropriate default arguments

# Load train and validation DataFrames from disk.
train = spark.read.load(train_df_path)
val = spark.read.load(val_df_path)

# Take a stratified sample.
p=0.01
train_sample = train.drop("__INDEX").sampleBy("tumor_score", fractions={1: p, 2: p, 3: p}, seed=42)
val_sample = val.drop("__INDEX").sampleBy("tumor_score", fractions={1: p, 2: p, 3: p}, seed=42)

# Reassign row indices.
# TODO: Wrap this in a function with appropriate default arguments.
train_sample = (
  train_sample.rdd
              .zipWithIndex()
              .map(lambda r: (r[1] + 1, *r[0]))
              .toDF(['__INDEX', 'slide_num', 'tumor_score', 'molecular_score', 'sample']))
train_sample = train_sample.select(train_sample["__INDEX"].astype("int"),
                                   train_sample.slide_num.astype("int"),
                                   train_sample.tumor_score.astype("int"),
                                   train_sample.molecular_score,
                                   train_sample["sample"])

val_sample = (
  val_sample.rdd
            .zipWithIndex()
            .map(lambda r: (r[1] + 1, *r[0]))
            .toDF(['__INDEX', 'slide_num', 'tumor_score', 'molecular_score', 'sample']))
val_sample = val_sample.select(val_sample["__INDEX"].astype("int"),
                               val_sample.slide_num.astype("int"),
                               val_sample.tumor_score.astype("int"),
                               val_sample.molecular_score,
                               val_sample["sample"])

# Save train and validation DataFrames.
tr_sample_filename = "train_{}_sample_{}{}.parquet".format(p, sample_size,
    "_grayscale" if grayscale else "")
val_sample_filename = "val_{}_sample_{}{}.parquet".format(p, sample_size,
    "_grayscale" if grayscale else "")
train_sample_path = os.path.join(save_folder, tr_sample_filename)
val_sample_path = os.path.join(save_folder, val_sample_filename)
save(train_sample, train_sample_path, sample_size, grayscale)
save(val_sample, val_sample_path, sample_size, grayscale)


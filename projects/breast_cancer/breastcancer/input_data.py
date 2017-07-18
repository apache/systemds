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
Data utilities for the TUPAC16 breast cancer project.

This is early, experimental code.

TODO: Cleanup & add proper comments to all functions.
"""
import multiprocessing as mp
import os
import threading

import numpy as np
import py4j


# Utils for reading data

def compute_channel_means(rdd, channels, size):
  """Compute the means of each color channel across the dataset."""
  # TODO: Replace this with pyspark.ml.feature.VectorSlicer
  # to cut vector into separate channel vectors, then grab the mean
  # of those new columns, all using DataFrame functions, rather than
  # RDD functions.
  # from pyspark.ml.linalg import VectorUDT
  # from pyspark.sql.functions import udf
  # from pyspark.sql import functions as F
  # as_ml = udf(lambda v: v.asML() if v is not None else None, VectorUDT())
  # slicers[0].transform(train_df.withColumn("sample", as_ml("sample"))).select(F.avg("ch0"))
  # slicers = [VectorSlicer(inputCol="sample", outputCol="ch{}".format(c), indices=range(c*pixels, c*pixels + pixels)) for c in range(CHANNELS)]
  def helper(x):
    x = x.sample.values
    x = np.array(x)
    x = (x.reshape((-1,channels,size,size))  # shape (N,C,H,W)
              .transpose((0,2,3,1))  # shape (N,H,W,C)
              .astype(np.float32))
    mu = np.mean(x, axis=(0,1,2))
    return mu

  means = rdd.map(helper).collect()
  means = np.array(means)
  means = np.mean(means, axis=0)
  return means


def gen_class_weights(df):
  """Generate class weights to even out the class distribution during training."""
  class_counts_df = df.select("tumor_score").groupBy("tumor_score").count()
  class_counts = {row["tumor_score"]:row["count"] for row in class_counts_df.collect()}
  max_count = max(class_counts.values())
  class_weights = {k-1:max_count/v for k,v in class_counts.items()}
  return class_weights


def read_data(spark_session, filename_template, sample_size, channels, sample_prob, normalize_class_distribution, seed):
  """Read and return training & validation Spark DataFrames."""
  # TODO: Clean this function up!!!
  assert channels in (1, 3)
  grayscale = False if channels == 3 else True

  # Sample (Optional)
  if sample_prob < 1:
    try:
      # Ex: `train_0.01_sample_256.parquet`
      sampled_filename_template = filename_template.format("{}_sample_".format(sample_prob), sample_size, "_grayscale" if grayscale else "")
      filename = os.path.join("data", sampled_filename_template)
      df = spark_session.read.load(filename)
    except:  # Pre-sampled DataFrame not available
      filename = os.path.join("data", filename_template.format("", sample_size, "_grayscale" if grayscale else ""))
      df = spark_session.read.load(filename)
      p = sample_prob  # sample percentage
      if normalize_class_distribution:
        # stratified sample with even class proportions
        n = df.count()  # num examples
        K = 3  # num classes
        s = p * n  # num examples in p% sample, as a fraction
        s_k = s / K  # num examples per class in evenly-distributed p% sample, as fraction
        class_counts_df = df.select("tumor_score").groupBy("tumor_score").count()
        class_counts = {row["tumor_score"]:row["count"] for row in class_counts_df.collect()}
        ps = {k:s_k/v for k,v in class_counts.items()}
        df = df.sampleBy("tumor_score", fractions=ps, seed=seed)
      else:
        # stratified sample maintaining the original class proportions
        df = df.sampleBy("tumor_score", fractions={1: p, 2: p, 3: p}, seed=seed)
      # TODO: Determine if coalesce actually provides a perf benefit on Spark 2.x
      #train_df.cache(), val_df.cache()  # cache here, or coalesce will hang
      # tc = train_df.count()
      # vc = val_df.count()
      #
      # # Reduce num partitions to ideal size (~128 MB/partition, determined empirically)
      # current_tr_parts = train_df.rdd.getNumPartitions()
      # current_val_parts = train_df.rdd.getNumPartitions()
      # ex_mb = sample_size * sample_size * channels * 8 / 1024 / 1024  # size of one example in MB
      # ideal_part_size_mb = 128  # 128 MB partitions sizes are empirically ideal
      # ideal_exs_per_part = round(ideal_part_size_mb / ex_mb)
      # tr_parts = round(tc / ideal_exs_per_part)
      # val_parts = round(vc / ideal_exs_per_part)
      # if current_tr_parts > tr_parts:
      #   train_df = train_df.coalesce(tr_parts)
      # if current_val_parts > val_parts:
      #   val_df = val_df.coalesce(val_parts)
      # train_df.cache(), val_df.cache()
  else:
    # Read in data
    filename = os.path.join("data", filename_template.format("", sample_size, "_grayscale" if grayscale else ""))
    df = spark_session.read.load(filename)

  return df


def read_train_data(spark_session, sample_size, channels, sample_prob=1, normalize_class_distribution=False, seed=42):
  """Read training Spark DataFrame."""
  filename = "train_{}{}{}_updated.parquet"
  train_df = read_data(spark_session, filename, sample_size, channels, sample_prob, normalize_class_distribution, seed)
  return train_df


def read_val_data(spark_session, sample_size, channels, sample_prob=1, normalize_class_distribution=False, seed=42):
  """Read validation Spark DataFrame."""
  filename = "val_{}{}{}_updated.parquet"
  train_df = read_data(spark_session, filename, sample_size, channels, sample_prob, normalize_class_distribution, seed)
  return train_df


# Utils for creating asynchronous queuing batch generators
# TODO: Add comments to these functions

def fill_partition_num_queue(partition_num_queue, num_partitions, stop_event):
  partition_num_queue.cancel_join_thread()
  while not stop_event.is_set():
    for i in range(num_partitions):
      partition_num_queue.put(i)


def fill_partition_queue(partition_queue, partition_num_queue, rdd, stop_event):
  partition_queue.cancel_join_thread()
  while not stop_event.is_set():
    # py4j has some issues with imports with first starting.
    try:
      partition_num = partition_num_queue.get()
      partition = rdd.context.runJob(rdd, lambda x: x, [partition_num])
      partition_queue.put(partition)
    except (AttributeError, py4j.protocol.Py4JError, Exception) as err:
      print("error: {}".format(err))


def fill_row_queue(row_queue, partition_queue, stop_event):
  row_queue.cancel_join_thread()
  while not stop_event.is_set():
    rows = partition_queue.get()
    for row in rows:
      row_queue.put(row)


def gen_batch(row_queue, batch_size):
  while True:
    features = []
    labels = []
    for i in range(batch_size):
      row = row_queue.get()
      features.append(row.sample.values)
      labels.append(row.tumor_score)
    x_batch = np.array(features).astype(np.uint8)
    y_batch = np.array(labels).astype(np.uint8)
    yield x_batch, y_batch


def create_batch_generator(
    rdd, batch_size=32, num_partition_threads=32, num_row_processes=16,
    partition_num_queue_size=128, partition_queue_size=16, row_queue_size=2048):
  """
  Create a  multiprocess batch generator.

  This creates a generator that uses processes and threads to create a
  pipeline that asynchronously fetches data from Spark, filling a set
  of queues, while yielding batches.  The goal here is to amortize the
  time needed to fetch data from Spark so that downstream consumers
  are saturated.
  """
  #rdd.cache()
  partition_num_queue = mp.Queue(partition_num_queue_size)
  partition_queue = mp.Queue(partition_queue_size)
  row_queue = mp.Queue(row_queue_size)

  num_partitions = rdd.getNumPartitions()
  stop_event = mp.Event()

  partition_num_process = mp.Process(target=fill_partition_num_queue, args=(partition_num_queue, num_partitions, stop_event), daemon=True)
  partition_threads = [threading.Thread(target=fill_partition_queue, args=(partition_queue, partition_num_queue, rdd, stop_event), daemon=True) for _ in range(num_partition_threads)]
  row_processes = [mp.Process(target=fill_row_queue, args=(row_queue, partition_queue, stop_event), daemon=True) for _ in range(num_row_processes)]

  ps = [partition_num_process] + row_processes + partition_threads
  queues = [partition_num_queue, partition_queue, row_queue]

  for p in ps:
    p.start()

  generator = gen_batch(row_queue, batch_size)
  return generator, ps, queues, stop_event


def stop(processes, stop_event):
  """Stop queuing processes."""
  stop_event.set()
  for p in processes:
    if isinstance(p, mp.Process):
      p.terminate()
  mp.active_children()  # Use to join the killed processes above.


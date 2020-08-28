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

from pyspark import SparkContext

from slicing.base.Bucket import Bucket
from slicing.base.SparkNode import SparkNode
from slicing.base.top_k import Topk
from slicing.spark_modules import spark_utils
from slicing.spark_modules.spark_utils import approved_join_slice


def rows_mapper(row, buckets, loss_type):
    # filtered = dict(filter(lambda bucket: all(attr in row[1] for attr in bucket[1].attributes), buckets.items()))
    filtered = dict(filter(lambda bucket: all(attr in row[0] for attr in bucket[1].attributes), buckets.items()))
    for item in filtered:
        filtered[item].update_metrics(row, loss_type)
    return filtered


def join_enum(cur_lvl_nodes, cur_lvl, x_size, alpha, top_k, w, loss):
    buckets = {}
    for node_i in range(len(cur_lvl_nodes)):
        for node_j in range(node_i + 1, len(cur_lvl_nodes)):
            flag = approved_join_slice(cur_lvl_nodes[node_i], cur_lvl_nodes[node_j], cur_lvl)
            if flag:
                node = SparkNode(None, None)
                node.attributes = list(set(cur_lvl_nodes[node_i].attributes) | set(cur_lvl_nodes[node_j].attributes))
                bucket = Bucket(node, cur_lvl, w, x_size, loss)
                bucket.parents.append(cur_lvl_nodes[node_i])
                bucket.parents.append(cur_lvl_nodes[node_j])
                bucket.calc_bounds(w, x_size, loss)
                if bucket.check_bounds(x_size, alpha, top_k):
                    buckets[bucket.name] = bucket
    return buckets


def combiner(a):
    return a


def merge_values(a, b):
    a.combine_with(b)
    return a


def merge_combiners(a, b):
    a + b
    return a


def parallel_process(all_features, predictions, loss, sc, debug, alpha, k, w, loss_type):
    top_k = Topk(k)
    cur_lvl = 0
    cur_lvl_nodes = list(all_features)
    pred_pandas = predictions.toPandas()
    x_size = len(pred_pandas)
    b_topk = SparkContext.broadcast(sc, top_k)
    b_cur_lvl = SparkContext.broadcast(sc, cur_lvl)
    buckets = {}
    for node in cur_lvl_nodes:
        bucket = Bucket(node, cur_lvl, w, x_size, loss)
        buckets[bucket.name] = bucket
    b_buckets = SparkContext.broadcast(sc, buckets)
    rows = predictions.rdd.map(lambda row: (row[1].indices, row[2]))\
        .map(lambda item: list(item))
    mapped = rows.map(lambda row: rows_mapper(row, b_buckets.value, loss_type))
    flattened = mapped.flatMap(lambda line: (line.items()))
    reduced = flattened.combineByKey(combiner, merge_values, merge_combiners)
    cur_lvl_nodes = reduced.values()\
        .map(lambda bucket: spark_utils.calc_bucket_metrics(bucket, loss, w, x_size, b_cur_lvl.value))
    if debug:
        cur_lvl_nodes.map(lambda bucket: bucket.print_debug(b_topk.value)).collect()
    cur_lvl = 1
    prev_level = cur_lvl_nodes.collect()
    top_k = top_k.buckets_top_k(prev_level, x_size, alpha, 1)
    while len(prev_level) > 0:
        b_cur_lvl_nodes = SparkContext.broadcast(sc, prev_level)
        b_topk = SparkContext.broadcast(sc, top_k)
        cur_min = top_k.min_score
        b_cur_lvl = SparkContext.broadcast(sc, cur_lvl)
        top_k.print_topk()
        buckets = join_enum(prev_level, cur_lvl, x_size, alpha, top_k, w, loss)
        b_buckets = SparkContext.broadcast(sc, buckets)
        to_slice = dict(filter(lambda bucket: bucket[1].check_bounds(x_size, alpha, top_k), buckets.items()))
        b_to_slice = SparkContext.broadcast(sc, to_slice)
        mapped = rows.map(lambda row: rows_mapper(row, b_to_slice.value, loss_type))
        flattened = mapped.flatMap(lambda line: (line.items()))
        to_process = flattened.combineByKey(combiner, merge_values, merge_combiners)
        if debug:
            to_process.values().map(lambda bucket: bucket.print_debug(b_topk.value)).collect()
        prev_level = to_process\
            .map(lambda bucket: spark_utils.calc_bucket_metrics(bucket[1], loss, w, x_size, b_cur_lvl.value))\
            .collect()
        cur_lvl += 1
        top_k = top_k.buckets_top_k(prev_level, x_size, alpha, cur_min)
        print("Level " + str(cur_lvl) + " had " + str(
            len(b_cur_lvl_nodes.value * (len(prev_level) - 1)))+" candidates but after pruning only " +
              str(len(prev_level)) + " go to the next level")
        top_k.print_topk()
    print()
    print("Program stopped at level " + str(cur_lvl - 1))
    print("Selected slices are: ")
    top_k.print_topk()
    return None

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

from slicing.base.top_k import Topk
from slicing.spark_modules import spark_utils
from slicing.spark_modules.spark_utils import update_top_k


def process(all_features, predictions, loss, sc, debug, alpha, k, w, loss_type, enumerator):
    top_k = Topk(k)
    cur_lvl = 0
    levels = []
    all_features = list(all_features)
    first_level = {}
    first_tasks = sc.parallelize(all_features)
    b_topk = SparkContext.broadcast(sc, top_k)
    init_slices = first_tasks.mapPartitions(lambda features: spark_utils.make_first_level(features, predictions, loss,
                                                                                          b_topk.value, w, loss_type)) \
        .map(lambda node: (node.key, node)) \
        .collect()
    first_level.update(init_slices)
    update_top_k(first_level, top_k, alpha, predictions, 1)
    prev_level = SparkContext.broadcast(sc, first_level)
    levels.append(prev_level)
    cur_lvl = 1
    top_k.print_topk()
    while len(levels[cur_lvl - 1].value) > 0:
        cur_lvl_res = {}
        b_topk = SparkContext.broadcast(sc, top_k)
        cur_min = top_k.min_score
        for left in range(int(cur_lvl / 2) + 1):
            right = cur_lvl - left - 1
            partitions = sc.parallelize(levels[left].value.values())
            mapped = partitions.mapPartitions(lambda nodes: spark_utils.nodes_enum(nodes, levels[right].value.values(),
                                                                                   predictions, loss, b_topk.value, alpha, k,
                                                                                   w, loss_type, cur_lvl, debug,
                                                                                   enumerator, cur_min))
            flattened = mapped.flatMap(lambda node: node)
            partial = flattened.map(lambda node: (node.key, node)).collect()
            cur_lvl_res.update(partial)
        prev_level = SparkContext.broadcast(sc, cur_lvl_res)
        levels.append(prev_level)
        update_top_k(cur_lvl_res, top_k, alpha, predictions, cur_min)
        cur_lvl = cur_lvl + 1
        top_k.print_topk()
        print("Level " + str(cur_lvl) + " had " + str(len(levels[cur_lvl - 1].value) * (len(levels[cur_lvl - 1].value) - 1)) +
              " candidates but after pruning only " + str(len(prev_level.value)) + " go to the next level")
    print("Program stopped at level " + str(cur_lvl - 1))
    print()
    print("Selected slices are: ")
    top_k.print_topk()
    return top_k

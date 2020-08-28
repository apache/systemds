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

from slicing.base.SparkNode import SparkNode
from slicing.base.slicer import union, opt_fun
from slicing.base.top_k import Topk
from slicing.spark_modules import spark_utils
from slicing.spark_modules.spark_utils import update_top_k


def join_enum_fun(node_a, list_b, predictions, f_l2, debug, alpha, w, loss_type, cur_lvl, top_k):
    x_size = len(predictions)
    nodes = []
    for node_i in range(len(list_b)):
        flag = spark_utils.approved_join_slice(node_i, node_a, cur_lvl)
        if not flag:
            new_node = SparkNode(predictions, f_l2)
            parents_set = set(new_node.parents)
            parents_set.add(node_i)
            parents_set.add(node_a)
            new_node.parents = list(parents_set)
            parent1_attr = node_a.attributes
            parent2_attr = list_b[node_i].attributes
            new_node_attr = union(parent1_attr, parent2_attr)
            new_node.attributes = new_node_attr
            new_node.name = new_node.make_name()
            new_node.calc_bounds(cur_lvl, w)
            # check if concrete data should be extracted or not (only for those that have score upper
            # and if size of subset is big enough
            to_slice = new_node.check_bounds(top_k, x_size, alpha)
            if to_slice:
                new_node.process_slice(loss_type)
                new_node.score = opt_fun(new_node.loss, new_node.size, f_l2, x_size, w)
                # we decide to add node to current level nodes (in order to make new combinations
                # on the next one or not basing on its score value
                if new_node.check_constraint(top_k, x_size, alpha) and new_node.key not in top_k.keys:
                    top_k.add_new_top_slice(new_node)
                nodes.append(new_node)
            if debug:
                new_node.print_debug(top_k, cur_lvl)
    return nodes


def parallel_process(all_features, predictions, loss, sc, debug, alpha, k, w, loss_type, enumerator):
    top_k = Topk(k)
    cur_lvl = 0
    levels = []
    first_level = {}
    all_features = list(all_features)
    first_tasks = sc.parallelize(all_features)
    b_topk = SparkContext.broadcast(sc, top_k)
    init_slices = first_tasks.mapPartitions(lambda features: spark_utils.make_first_level(features, predictions, loss,
                                                                                          b_topk.value, w, loss_type)) \
        .map(lambda node: (node.key, node)).collect()
    first_level.update(init_slices)
    update_top_k(first_level, top_k, alpha, predictions, 1)
    prev_level = SparkContext.broadcast(sc, first_level)
    levels.append(prev_level)
    cur_lvl = cur_lvl + 1
    top_k.print_topk()
    # checking the first partition of level. if not empty then processing otherwise no elements were added to this level
    while len(levels[cur_lvl - 1].value) > 0:
        nodes_list = {}
        b_topk = SparkContext.broadcast(sc, top_k)
        cur_min = top_k.min_score
        partitions = sc.parallelize(levels[cur_lvl - 1].value.values())
        mapped = partitions.mapPartitions(lambda nodes: spark_utils.nodes_enum(nodes, levels[cur_lvl - 1].value.values(),
                                                                               predictions, loss, b_topk.value, alpha, k, w,
                                                                               loss_type, cur_lvl, debug, enumerator, cur_min))
        flattened = mapped.flatMap(lambda node: node)
        nodes_list.update(flattened.map(lambda node: (node.key, node)).distinct().collect())
        prev_level = SparkContext.broadcast(sc, nodes_list)
        levels.append(prev_level)
        update_top_k(nodes_list, top_k, alpha, predictions, cur_min)
        cur_lvl = cur_lvl + 1
        b_topk.value.print_topk()
        print("Level " + str(cur_lvl) + " had " + str(len(levels[cur_lvl - 1].value) * (len(levels[cur_lvl - 1].value) - 1)) +
              " candidates but after pruning only " + str(len(nodes_list)) + " go to the next level")
    print("Program stopped at level " + str(cur_lvl - 1))
    print()
    print("Selected slices are: ")
    top_k.print_topk()
    return top_k

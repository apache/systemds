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


from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

from slicing.base.SparkNode import SparkNode
from slicing.base.slicer import opt_fun, union

calc_loss = udf(lambda target, prediction, type: calc_loss_fun(target, prediction, type), FloatType())
model_type_init = udf(lambda type: init_model_type(type))


def calc_loss_fun(target, prediction, type):
    if type == 0:
        return (prediction - target) ** 2
    elif type == 1:
        return float(target == prediction)


def init_model_type(model_type):
    if model_type == "regression":
        return 0
    elif model_type == "classification":
        return 1


def approved_join_slice(node_i, node_j, cur_lvl):
    commons = len(list(set(node_i.attributes) & set(node_j.attributes)))
    return commons == cur_lvl - 1


def approved_union_slice(node_i, node_j):
    if set(node_i.attributes).intersection(set(node_j.attributes)):
        return False
    return True


def make_first_level(features, predictions, loss, top_k, w, loss_type):
    first_level = []
    # First level slices are enumerated in a "classic way" (getting data and not analyzing bounds
    for feature in features:
        new_node = SparkNode(loss, predictions)
        new_node.parents = [feature]
        new_node.attributes.append(feature)
        new_node.name = new_node.make_name()
        new_node.key = new_node.make_key()
        new_node.process_slice(loss_type)
        new_node.score = opt_fun(new_node.loss, new_node.size, loss, len(predictions), w)
        new_node.c_upper = new_node.score
        first_level.append(new_node)
        new_node.print_debug(top_k, 0)
    return first_level


def process_node(node_i, level, loss, predictions, cur_lvl, top_k, alpha, loss_type, w, debug, enumerator):
    cur_enum_nodes = []
    for node_j in level:
        if enumerator == "join":
            flag = approved_join_slice(node_i, node_j, cur_lvl)
        else:
            flag = approved_union_slice(node_i, node_j)
        if flag and int(node_i.name.split("&&")[0]) < int(node_j.name.split("&&")[0]):
            new_node = SparkNode(loss, predictions)
            parents_set = set(new_node.parents)
            parents_set.add(node_i)
            parents_set.add(node_j)
            new_node.parents = list(parents_set)
            parent1_attr = node_i.attributes
            parent2_attr = node_j.attributes
            new_node_attr = union(parent1_attr, parent2_attr)
            new_node.attributes = new_node_attr
            new_node.name = new_node.make_name()
            new_node.key = new_node.make_key()
            new_node.calc_bounds(cur_lvl, w)
            to_slice = new_node.check_bounds(top_k, len(predictions), alpha)
            if to_slice:
                new_node.process_slice(loss_type)
                new_node.score = opt_fun(new_node.loss, new_node.size, loss, len(predictions), w)
                if new_node.check_constraint(top_k, len(predictions), alpha):
                    cur_enum_nodes.append(new_node)
            if debug:
                new_node.print_debug(top_k, cur_lvl)
    return cur_enum_nodes


def nodes_enum(nodes, level, predictions, loss, top_k, alpha, k, w, loss_type, cur_lvl, debug, enumerator):
    cur_enum_nodes = []
    for node_i in nodes:
        partial_nodes = process_node(node_i, level, loss, predictions, cur_lvl, top_k, alpha,
                                     loss_type, w, debug, enumerator)
        cur_enum_nodes.append(partial_nodes)
    return cur_enum_nodes


def init_top_k(first_level, top_k, alpha, predictions):
    # driver updates topK
    for sliced in first_level:
        if sliced[1].check_constraint(top_k, len(predictions), alpha):
            # this method updates top k slices if needed
            top_k.add_new_top_slice(sliced[1])


def update_top_k(new_slices, top_k, alpha, predictions):
    # driver updates topK
    for sliced in new_slices.values():
        if sliced.check_constraint(top_k, len(predictions), alpha):
            # this method updates top k slices if needed
            top_k.add_new_top_slice(sliced)


def calc_bucket_metrics(bucket, loss, w, x_size, cur_lvl):
    bucket.calc_error()
    bucket.score = opt_fun(bucket.error, bucket.size, loss, x_size, w)
    if cur_lvl == 0:
        bucket.s_upper = bucket.size
        bucket.c_upper = bucket.score
        bucket.s_lower = 1
    return bucket

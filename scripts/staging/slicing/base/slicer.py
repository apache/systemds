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

from slicing.base.node import Node
from slicing.base.top_k import Topk


# optimization function calculation:
# fi - error of subset, si - its size
# f - error of complete set, x_size - its size
# w - "significance" weight of error constraint
def opt_fun(fi, si, f, x_size, w):
    formula = w * (fi/f) + (1 - w) * (si/x_size)
    return float(formula)


# slice_name_nonsense function defines if combination of nodes on current level is fine or impossible:
# there is dependency between common nodes' attributes number and current level is such that:
# commons == cur_lvl - 1
# valid combination example: node ABC + node BCD (on 4th level) // three attributes nodes have two common attributes
# invalid combination example: node ABC + CDE (on 4th level) // result node - ABCDE (absurd for 4th level)
def slice_name_nonsense(node_i, node_j, cur_lvl):
    attr1 = list(map(lambda x: x[0].split("_")[0], node_i.attributes))
    attr2 = list(map(lambda x: x[0].split("_")[0], node_j.attributes))
    commons = len(list(set(attr1) & set(attr2)))
    return commons == cur_lvl - 1


def union(lst1, lst2):
    final_list = sorted(set(list(lst1) + list(lst2)))
    return final_list


def make_first_level(all_features, complete_x, loss, x_size, y_test, errors, loss_type, top_k, alpha, w):
    first_level = []
    counter = 0
    all_nodes = {}
    # First level slices are enumerated in a "classic way" (getting data and not analyzing bounds
    for feature in all_features:
        new_node = Node(complete_x, loss, x_size, y_test, errors)
        new_node.parents = [(feature, counter)]
        new_node.attributes.append((feature, counter))
        new_node.name = new_node.make_name()
        new_id = len(all_nodes)
        new_node.key = new_node.make_key(new_id)
        all_nodes[new_node.key] = new_node
        new_node.process_slice(loss_type)
        new_node.score = opt_fun(new_node.loss, new_node.size, loss, x_size, w)
        new_node.c_upper = new_node.score
        first_level.append(new_node)
        new_node.print_debug(top_k, 0)
        # constraints for 1st level nodes to be problematic candidates
        if new_node.check_constraint(top_k, x_size, alpha):
            # this method updates top k slices if needed
            top_k.add_new_top_slice(new_node)
        counter = counter + 1
    return first_level, all_nodes


def join_enum(node_i, prev_lvl, complete_x, loss, x_size, y_test, errors, debug, alpha, w, loss_type, b_update, cur_lvl,
              all_nodes, top_k, cur_lvl_nodes):
    for node_j in range(len(prev_lvl)):
        flag = slice_name_nonsense(prev_lvl[node_i], prev_lvl[node_j], cur_lvl)
        if flag and prev_lvl[node_j].key[0] > prev_lvl[node_i].key[0]:
            new_node = Node(complete_x, loss, x_size, y_test, errors)
            parents_set = set(new_node.parents)
            parents_set.add(prev_lvl[node_i])
            parents_set.add(prev_lvl[node_j])
            new_node.parents = list(parents_set)
            parent1_attr = prev_lvl[node_i].attributes
            parent2_attr = prev_lvl[node_j].attributes
            new_node_attr = union(parent1_attr, parent2_attr)
            new_node.attributes = new_node_attr
            new_node.name = new_node.make_name()
            new_id = len(all_nodes)
            new_node.key = new_node.make_key(new_id)
            if new_node.key[1] in all_nodes:
                existing_item = all_nodes[new_node.key[1]]
                parents_set = set(existing_item.parents)
                existing_item.parents = parents_set
                if b_update:
                    s_upper = new_node.calc_s_upper(cur_lvl)
                    s_lower = new_node.calc_s_lower(cur_lvl)
                    e_upper = new_node.calc_e_upper()
                    e_max_upper = new_node.calc_e_max_upper(cur_lvl)
                    new_node.update_bounds(s_upper, s_lower, e_upper, e_max_upper, w)
            else:
                new_node.calc_bounds(cur_lvl, w)
                all_nodes[new_node.key[1]] = new_node
                # check if concrete data should be extracted or not (only for those that have score upper
                # big enough and if size of subset is big enough
                to_slice = new_node.check_bounds(top_k, x_size, alpha)
                if to_slice:
                    new_node.process_slice(loss_type)
                    new_node.score = opt_fun(new_node.loss, new_node.size, loss, x_size, w)
                    # we decide to add node to current level nodes (in order to make new combinations
                    # on the next one or not basing on its score value
                    if new_node.check_constraint(top_k, x_size, alpha) and new_node.key not in top_k.keys:
                        top_k.add_new_top_slice(new_node)
                    cur_lvl_nodes.append(new_node)
                if debug:
                    new_node.print_debug(top_k, cur_lvl)
    return cur_lvl_nodes, all_nodes


# alpha is size significance coefficient (required for optimization function)
# verbose option is for returning debug info while creating slices and printing it (in console)
# k is number of top-slices we want to receive as a result (maximum output, if less all of subsets will be printed)
# w is a weight of error function significance (1 - w) is a size significance propagated into optimization function
# loss_type = 0 (in case of regression model)
# loss_type = 1 (in case of classification model)
def process(all_features, complete_x, loss, x_size, y_test, errors, debug, alpha, k, w, loss_type, b_update):
    levels = []
    top_k = Topk(k)
    first_level = make_first_level(all_features, complete_x, loss, x_size, y_test, errors, loss_type, top_k, alpha, w)
    all_nodes = first_level[1]
    levels.append(first_level[0])
    # cur_lvl - index of current level, correlates with number of slice forming features
    cur_lvl = 1  # currently filled level after first init iteration
    # currently for debug
    print("Level 1 had " + str(len(all_features)) + " candidates")
    print()
    print("Current topk are: ")
    top_k.print_topk()
    # combining each candidate of previous level with every till it becomes useless (one node can't make a pair)
    while len(levels[cur_lvl - 1]) > 1:
        cur_lvl_nodes = []
        prev_lvl = levels[cur_lvl - 1]
        for node_i in range(len(prev_lvl)):
            partial = join_enum(node_i, prev_lvl, complete_x, loss, x_size, y_test, errors, debug, alpha, w, loss_type,
                                b_update, cur_lvl, all_nodes, top_k, cur_lvl_nodes)
            cur_lvl_nodes = partial[0]
            all_nodes = partial[1]
        cur_lvl = cur_lvl + 1
        levels.append(cur_lvl_nodes)
        top_k.print_topk()
        print("Level " + str(cur_lvl) + " had " + str(len(prev_lvl) * (len(prev_lvl) - 1)) +
              " candidates but after pruning only " + str(len(cur_lvl_nodes)) + " go to the next level")
    print("Program stopped at level " + str(cur_lvl + 1))
    print()
    print("Selected slices are: ")
    top_k.print_topk()

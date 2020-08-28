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
from slicing.base.slicer import opt_fun, union
import matplotlib.pyplot as plt


def check_attributes(left_node, right_node):
    attr1 = list(map(lambda x: x[0].split("_")[0], left_node.attributes))
    attr2 = list(map(lambda x: x[0].split("_")[0], right_node.attributes))
    if set(attr1).intersection(set(attr2)):
        return False
    return True


def make_first_level(all_features, complete_x, loss, x_size, y_test, errors, loss_type, w, alpha, top_k):
    all_nodes = {}
    counter = 0
    first_level = []
    for feature in all_features:
        new_node = Node(complete_x, loss, x_size, y_test, errors)
        new_node.parents = [(feature, counter)]
        new_node.attributes.append((feature, counter))
        new_node.name = new_node.make_name()
        new_id = len(all_nodes)
        new_node.key = new_node.make_key(new_id)
        all_nodes[new_node.key] = new_node
        new_node.process_slice(loss_type)
        # for first level nodes all bounds are strict as concrete metrics
        new_node.s_upper = new_node.size
        new_node.s_lower = 0
        new_node.e_upper = new_node.loss
        new_node.e_max_upper = new_node.e_max
        new_node.score = opt_fun(new_node.loss, new_node.size, loss, x_size, w)
        new_node.c_upper = new_node.score
        first_level.append(new_node)
        new_node.print_debug(top_k, 0)
        # constraints for 1st level nodes to be problematic candidates
        if new_node.score > 1 and new_node.size >= x_size / alpha:
            # this method updates top k slices if needed
            top_k.add_new_top_slice(new_node)
        counter = counter + 1
    return first_level, all_nodes


def process(all_features, complete_x, loss, x_size, y_test, errors, debug, alpha, k, w, loss_type, b_update):
    top_k = Topk(k)
    # First level slices are enumerated in a "classic way" (getting data and not analyzing bounds
    levels = []
    first_level = make_first_level(all_features, complete_x, loss, x_size, y_test, errors, loss_type, w, alpha, top_k)
    # double appending of first level nodes in order to enumerating second level in the same way as others
    levels.append((first_level[0], len(all_features)))
    candidates = []
    pruned = []
    indexes = []
    indexes.append(1)
    candidates.append(len(first_level[0]))
    pruned.append(len(first_level[0]))
    all_nodes = first_level[1]
    # cur_lvl - index of current level, correlates with number of slice forming features
    cur_lvl = 1  # level that is planned to be filled later
    cur_lvl_nodes = first_level
    # currently for debug
    print("Level 1 had " + str(len(all_features)) + " candidates")
    print()
    print("Current topk are: ")
    top_k.print_topk()
    # DPSize algorithm approach of previous levels nodes combinations and updating bounds for those that already exist
    while len(cur_lvl_nodes) > 0:
        cur_lvl_nodes = []
        count = 0
        prev_lvl = levels[cur_lvl - 1]
        for left in range(int(cur_lvl / 2) + 1):
            right = cur_lvl - 1 - left
            for node_i in range(len(levels[left][0])):
                for node_j in range(len(levels[right][0])):
                    flag = check_attributes(levels[left][0][node_i], levels[right][0][node_j])
                    required_number = len(union(levels[left][0][node_i].attributes, levels[right][0][node_j].attributes))
                    if flag and required_number == cur_lvl + 1:
                        new_node = Node(complete_x, loss, x_size, y_test, errors)
                        parents_set = set(new_node.parents)
                        parents_set.add(levels[left][0][node_i])
                        parents_set.add(levels[right][0][node_j])
                        new_node.parents = list(parents_set)
                        parent1_attr = levels[left][0][node_i].attributes
                        parent2_attr = levels[right][0][node_j].attributes
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
                            to_slice = new_node.check_bounds(x_size, alpha, top_k.min_score)
                            if to_slice:
                                new_node.process_slice(loss_type)
                                new_node.score = opt_fun(new_node.loss, new_node.size, loss, x_size, w)
                                # we decide to add node to current level nodes (in order to make new combinations
                                # on the next one or not basing on its score value
                                if new_node.check_constraint(top_k, x_size, alpha, top_k.min_score) and new_node.key \
                                        not in top_k.keys:
                                    top_k.add_new_top_slice(new_node)
                                cur_lvl_nodes.append(new_node)
                            if debug:
                                new_node.print_debug(top_k, cur_lvl)
            count = count + levels[left][1] * levels[right][1]
        cur_lvl = cur_lvl + 1
        indexes.append(cur_lvl)
        print("Level " + str(cur_lvl) + " had " + str(count) +
              " candidates but after pruning only " + str(len(cur_lvl_nodes)) + " go to the next level")
        levels.append((cur_lvl_nodes, count))
        candidates.append(count)
        pruned.append(len(cur_lvl_nodes))
        top_k.print_topk()
    plt.plot(indexes, candidates, 'r--',
             indexes, pruned, 'g--')
    plt.xlabel('Level')
    plt.ylabel('Number of slices')
    plt.show()
    print("Program stopped at level " + str(cur_lvl))
    print()
    print("Selected slices are: ")
    top_k.print_topk()
    print("candidates:")
    print(candidates)
    print(">>>>>>>>>")
    print("pruned:")
    print(pruned)
    return top_k

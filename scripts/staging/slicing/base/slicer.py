#-------------------------------------------------------------
#
# Copyright 2020 Graz University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#-------------------------------------------------------------

from slicing.node import Node
from slicing.top_k import Topk


# optimization function calculation:
# fi - error of subset, si - its size
# f - error of complete set, x_size - its size
# w - "significance" weight of error constraint
def opt_fun(fi, si, f, x_size, w):
    formula = w * (fi/f) + (1 - w) * (si/x_size)
    return float(formula)


# slice_name_nonsense function defines if combination of nodes on current level is fine or impossible:
# there is dependency between common nodes' common attributes number and current level is such that:
# commons == cur_lvl - 1
# valid combination example: node ABC + node BCD (on 4th level) // three attributes nodes have two common attributes
# invalid combination example: node ABC + CDE (on 4th level) // result node - ABCDE (absurd for 4th level)
def slice_name_nonsense(node_i, node_j, cur_lvl):
    commons = 0
    for attr1 in node_i.attributes:
        for attr2 in node_j.attributes:
            if attr1[0].split("_")[0] == attr2[0].split("_")[0]:
                commons = commons + 1
    return commons != cur_lvl - 1


def union(lst1, lst2):
    final_list = sorted(set(list(lst1) + list(lst2)))
    return final_list


def check_for_slicing(node, topk, x_size, alpha):
    return node.s_upper >= x_size / alpha and node.c_upper >= topk.min_score


# alpha is size significance coefficient (required for optimization function)
# verbose option is for returning debug info while creating slices and printing it (in console)
# k is number of top-slices we want to receive as a result (maximum output, if less all of subsets will be printed)
# w is a weight of error function significance (1 - w) is a size significance propagated into optimization function
# loss_type = 0 (in case of regression model)
# loss_type = 1 (in case of classification model)
def process(all_features, model, complete_x, loss, x_size, y_test, errors, debug, alpha, k, w, loss_type):
    counter = 0
    # First level slices are enumerated in a "classic way" (getting data and not analyzing bounds
    first_level = []
    levels = []
    all_nodes = {}
    top_k = Topk(k)
    for feature in all_features:
        new_node = Node(all_features, model, complete_x, loss, x_size, y_test, errors)
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
        if new_node.score > 1 and new_node.size >= x_size / alpha:
            # this method updates top k slices if needed
            top_k.add_new_top_slice(new_node)
        counter = counter + 1
    levels.append(first_level)

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
        for node_i in range(len(levels[cur_lvl - 1]) - 1):
            for node_j in range(node_i + 1, len(levels[cur_lvl - 1])):
                flag = slice_name_nonsense(levels[cur_lvl - 1][node_i], levels[cur_lvl - 1][node_j], cur_lvl)
                if not flag:
                    new_node = Node(all_features, model, complete_x, loss, x_size, y_test, errors)
                    parents_set = set(new_node.parents)
                    parents_set.add(levels[cur_lvl - 1][node_i])
                    parents_set.add(levels[cur_lvl - 1][node_j])
                    new_node.parents = list(parents_set)
                    parent1_attr = levels[cur_lvl - 1][node_i].attributes
                    parent2_attr = levels[cur_lvl - 1][node_j].attributes
                    new_node_attr = union(parent1_attr, parent2_attr)
                    new_node.attributes = new_node_attr
                    new_node.name = new_node.make_name()
                    new_id = len(all_nodes)
                    new_node.key = new_node.make_key(new_id)
                    if new_node.key[1] in all_nodes:
                        existing_item = all_nodes[new_node.key[1]]
                        parents_set = set(existing_item.parents)
                        existing_item.parents = parents_set
                    else:
                        new_node.calc_bounds(cur_lvl, w)
                        all_nodes[new_node.key[1]] = new_node
                        # check if concrete data should be extracted or not (only for those that have score upper
                        # big enough and if size of subset is big enough
                        to_slice = check_for_slicing(new_node, top_k, x_size, alpha)
                        if to_slice:
                            new_node.process_slice(loss_type)
                            new_node.score = opt_fun(new_node.loss, new_node.size, loss, x_size, w)
                            # we decide to add node to current level nodes (in order to make new combinations
                            # on the next one or not basing on its score value
                            if new_node.score >= top_k.min_score:
                                cur_lvl_nodes.append(new_node)
                                top_k.add_new_top_slice(new_node)
                        if debug:
                            new_node.print_debug(top_k, cur_lvl)
        print("Level " + str(cur_lvl + 1) + " had " + str(len(levels[cur_lvl - 1]) * (len(levels[cur_lvl - 1]) - 1)) +
              " candidates but after pruning only " + str(len(cur_lvl_nodes)) + " go to the next level")
        cur_lvl = cur_lvl + 1
        levels.append(cur_lvl_nodes)
        top_k.print_topk()
    print("Program stopped at level " + str(cur_lvl + 1))
    print()
    print("Selected slices are: ")
    top_k.print_topk()


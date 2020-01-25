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


def opt_fun(fi, si, f, x_size, w):
    formula = w * (fi/f) + (1 - w) * (si/x_size)
    return float(formula)


def slice_name_nonsense(node_i, node_j, cur_lvl):
    commons = 0
    for attr1 in node_i.attributes:
        for attr2 in node_j.attributes:
            if attr1 == attr2:
                commons = commons + 1
    return commons != cur_lvl - 1


def union(lst1, lst2):
    final_list = sorted(set(list(lst1) + list(lst2)))
    return final_list


def check_for_slicing(node, w, topk, x_size, alpha):
    return node.s_upper >= x_size / alpha and node.eval_c_upper(w) >= topk.min_score


def check_for_excluding(node, topk, x_size, alpha, w):
    return node.s_upper >= x_size / alpha or node.eval_c_upper(w) >= topk.min_score


def process(enc, model, complete_x, complete_y, f_l2, x_size, x_test, y_test, debug, alpha, k, w):
    # forming pairs of encoded features
    all_features = enc.get_feature_names()
    features_indexes = []
    counter = 0
    # First level slices are enumerated in a "classic way" (getting data and not analyzing bounds
    first_level = []
    levels = []
    all_nodes = {}
    top_k = Topk(k)
    for feature in all_features:
        features_indexes.append((feature, counter))
        new_node = Node(enc, model, complete_x, complete_y, f_l2, x_size, x_test, y_test)
        new_node.parents = [(feature, counter)]
        new_node.attributes.append((feature, counter))
        new_node.name = new_node.make_name()
        new_id = len(all_nodes)
        new_node.key = new_node.make_key(new_id)
        all_nodes[new_node.key] = new_node
        subset = new_node.make_slice()
        new_node.size = len(subset)
        fi_l2 = 0
        tuple_errors = []
        for j in range(0, len(subset)):
            fi_l2_sample = (model.predict(subset[j][1].reshape(1, -1)) -
                            complete_y[subset[j][0]][1]) ** 2
            tuple_errors.append(fi_l2_sample)
            fi_l2 = fi_l2 + fi_l2_sample
        new_node.e_max = max(tuple_errors)
        new_node.l2_error = fi_l2 / new_node.size
        new_node.score = opt_fun(new_node.l2_error, new_node.size, f_l2, x_size, w)
        new_node.e_upper = max(tuple_errors)
        new_node.c_upper = new_node.score
        first_level.append(new_node)
        counter = counter + 1
    levels.append(first_level)
    candidates = []
    for sliced in first_level:
        if sliced.score > 1 and sliced.size >= x_size / alpha:
            candidates.append(sliced)
    # cur_lvl - index of current level, correlates with number of slice forming features
    cur_lvl = 1  # currently filled level after first init iteration
    for candidate in candidates:
        top_k.add_new_top_slice(candidate)
    while cur_lvl < len(all_features):
        cur_lvl_nodes = []
        for node_i in range(len(levels[cur_lvl - 1]) - 1):
            for node_j in range(node_i + 1, len(levels[cur_lvl - 1]) - 1):
                flag = slice_name_nonsense(levels[cur_lvl - 1][node_i], levels[cur_lvl - 1][node_j], cur_lvl)
                if not flag:
                    new_node = Node(enc, model, complete_x, complete_y, f_l2, x_size, x_test, y_test)
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
                    if new_node.key in all_nodes:
                        existing_item = all_nodes[new_node.key]
                        parents_set = set(existing_item.parents)
                        parents_set.add(node_i)
                        parents_set.add(node_j)
                        existing_item.parents = parents_set
                    else:
                        new_node.s_upper = new_node.calc_s_upper(cur_lvl)
                        new_node.s_lower = new_node.calc_s_lower(cur_lvl)
                        new_node.e_upper = new_node.calc_e_upper()
                        new_node.e_max_upper = new_node.calc_e_max_upper(cur_lvl)
                        new_node.c_upper = new_node.calc_c_upper(w)
                        all_nodes[new_node.key] = new_node
                        to_slice = check_for_slicing(new_node, w, top_k, x_size, alpha)
                        # we make data slicing basing on score upper bound
                        if to_slice:
                            subset = new_node.make_slice()
                            new_node.size = len(subset)
                            new_node.l2_error = new_node.calc_l2_error(subset)
                            new_node.score = opt_fun(new_node.l2_error, new_node.size, f_l2, x_size, w)
                            # we decide to add node to current level nodes (in order to make new combinations
                            # on the next one or not basing on its score value calculated according to actual size and
                            # L2 error of a sliced subset
                            if new_node.score >= top_k.min_score:
                                cur_lvl_nodes.append(new_node)
                                top_k.add_new_top_slice(new_node)
                    if debug:
                        new_node.print_debug(top_k, cur_lvl)
        cur_lvl = cur_lvl + 1
        levels.append(cur_lvl_nodes)
    top_k.print_topk()


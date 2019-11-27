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
from slicing.slicer import opt_fun, union


def check_attributes(left_node, right_node):
    flag = False
    for attr1 in left_node.attributes:
        for attr2 in right_node.attributes:
            if attr1[0].split("_")[0] == attr2[0].split("_")[0]:
                # there are common attributes which is not the case we need
                flag = True
                break
    return flag


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
    # double appending of first level nodes in order to enumerating second level in the same way as others
    levels.append((first_level, len(all_features)))
    levels.append((first_level, len(all_features)))

    # cur_lvl - index of current level, correlates with number of slice forming features
    cur_lvl = 2  # level that is planned to be filled later
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
        for left in range(int(cur_lvl / 2)):
            right = cur_lvl - 1 - left
            for node_i in range(len(levels[left][0])):
                for node_j in range(len(levels[right][0])):
                    flag = check_attributes(levels[left][0][node_i], levels[right][0][node_j])
                    if not flag:
                        new_node = Node(all_features, model, complete_x, loss, x_size, y_test, errors)
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
                            s_upper = new_node.calc_s_upper(cur_lvl)
                            s_lower = new_node.calc_s_lower(cur_lvl)
                            e_upper = new_node.calc_e_upper()
                            e_max_upper = new_node.calc_e_max_upper(cur_lvl)
                            try:
                                minimized = min(s_upper, new_node.s_upper)
                                new_node.s_upper = minimized
                                minimized = min(s_lower, new_node.s_lower)
                                new_node.s_lower = minimized
                                minimized = min(e_upper, new_node.e_upper)
                                new_node.e_upper = minimized
                                minimized= min(e_max_upper, new_node.e_max_upper)
                                new_node.e_max_upper = minimized
                                c_upper = new_node.calc_c_upper(w)
                                minimized= min(c_upper, new_node.c_upper)
                                new_node.c_upper = minimized
                            except AttributeError:
                                # initial bounds calculation
                                new_node.s_upper = s_upper
                                new_node.s_lower = s_lower
                                new_node.e_upper = e_upper
                                new_node.e_max_upper = e_max_upper
                                c_upper = new_node.calc_c_upper(w)
                                new_node.c_upper = c_upper
                            minimized = min(c_upper, new_node.c_upper)
                            new_node.c_upper = minimized
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
                                if new_node.score >= top_k.min_score and new_node.size >= x_size / alpha \
                                        and new_node.key not in top_k.keys:
                                    cur_lvl_nodes.append(new_node)
                                    top_k.add_new_top_slice(new_node)
                            else:
                                if new_node.s_upper >= x_size / alpha and new_node.c_upper >= top_k.min_score:
                                    cur_lvl_nodes.append(new_node)
                            if debug:
                                new_node.print_debug(top_k, cur_lvl)
            count = count + levels[left][1] * levels[right][1]
        print("Level " + str(cur_lvl) + " had " + str(count) +
              " candidates but after pruning only " + str(len(cur_lvl_nodes)) + " go to the next level")
        cur_lvl = cur_lvl + 1
        levels.append((cur_lvl_nodes, count))
        top_k.print_topk()
    print("Program stopped at level " + str(cur_lvl))
    print()
    print("Selected slices are: ")
    top_k.print_topk()

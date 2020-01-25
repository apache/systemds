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

def logic_comparator(list_a, list_b, attributes):
    result = [int(x) == y for (x, y) in zip(list_a, list_b)]
    counter = 0
    for item in result:
        if item:
            counter = counter + 1
    return counter == attributes


class Node:
    name: ""
    attributes: []
    parents: []
    children: []
    size: int
    l2_error: float
    score: float
    e_max: float
    s_upper: int
    s_lower: int
    e_upper: float
    c_upper: float
    e_max_upper: float

    def __init__(self, enc, model, complete_x, complete_y, f_l2, x_size, x_test, y_test):
        self.parents = []
        self.attributes = []
        self.size = 0
        self.score = 0
        self.enc = enc
        self.model = model
        self.complete_x = complete_x
        self.complete_y = complete_y
        self.f_l2 = f_l2
        self.x_size = x_size
        self.x_test = x_test
        self.y_test = y_test
        self.all_features = enc.get_feature_names()

    def calc_c_upper(self, w):
        upper_score = w * (self.e_upper / self.s_lower) / (self.f_l2 / self.x_size) + (1 - w) * self.s_upper
        return float(upper_score)

    def eval_c_upper(self, w):
        upper_score = w * (self.e_max_upper / (self.f_l2 / self.x_size)) + (1 - w) * self.s_upper
        return upper_score

    def make_slice_mask(self):
        mask = []
        attributes_indexes = []
        for feature in self.attributes:
            attributes_indexes.append(feature[1])
        i = 0
        while i < len(list(self.all_features)):
            if i in attributes_indexes:
                mask.append(1)
            else:
                mask.append(-1)
            i = i + 1
        return mask

    def make_slice(self):
        mask = self.make_slice_mask()
        subset = list(filter(lambda row: logic_comparator(list(row[1]), mask, len(self.attributes)) == True,
                                                                                            self.complete_x))
        return subset

    def calc_s_upper(self, cur_lvl):
        cur_min = self.parents[0].size
        for parent in self.parents:
            if cur_lvl == 1:
                cur_min = min(cur_min, parent.size)
            else:
                cur_min = min(cur_min, parent.s_upper)
        return cur_min

    def calc_e_max_upper(self, cur_lvl):
        if cur_lvl == 1:
            e_max_min = self.parents[0].e_max
        else:
            e_max_min = self.parents[0].e_max_upper
        for parent in self.parents:
            if cur_lvl == 1:
                e_max_min = min(e_max_min, parent.e_max)
            else:
                e_max_min = min(e_max_min, parent.e_max_upper)
        return e_max_min

    def calc_s_lower(self, cur_lvl):
        size_value = self.x_size
        for parent in self.parents:
            if cur_lvl == 1:
                size_value = size_value - (self.x_size - parent.size)
            else:
                size_value = size_value - (self.x_size - parent.s_lower)
        return max(size_value, 1)

    def calc_e_upper(self):
        prev_e_uppers = []
        for parent in self.parents:
            prev_e_uppers.append(parent.e_upper)
        return float(min(prev_e_uppers))

    def make_name(self):
        name = ""
        for attribute in self.attributes:
            name = name + str(attribute[0]) + " && "
        name = name[0: len(name) - 4]
        return name

    def make_key(self, new_id):
        return new_id, self.name

    def calc_l2_error(self, subset):
        fi_l2 = 0
        for i in range(0, len(subset)):
            fi_l2_sample = (self.model.predict(subset[i][1].reshape(1, -1)) - self.complete_y[subset[i][0]][1]) ** 2
            fi_l2 = fi_l2 + float(fi_l2_sample)
        if len(subset) > 0:
            fi_l2 = fi_l2 / len(subset)
        else:
            fi_l2 = 0
        return float(fi_l2)

    def print_debug(self, topk, level):
        print("new node has been added: " + self.make_name() + "\n")
        if level >= 1:
            print("s_upper = " + str(self.s_upper))
            print("s_lower = " + str(self.s_lower))
            print("e_upper = " + str(self.e_upper))
            print("c_upper = " + str(self.c_upper))
        print("size = " + str(self.size))
        print("score = " + str(self.score))
        print("current topk min score = " + str(topk.min_score))
        print("-------------------------------------------------------------------------------------")


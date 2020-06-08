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

class Bucket:

    key: []
    attributes: []
    name: ""
    error: float
    e_upper: float
    size: float
    sum_error: float
    max_tuple_error: float
    s_upper: float
    s_lower: float
    e_max: float
    e_max_upper: float
    score: float
    c_upper: float
    parents: []
    x_size: int
    loss: float
    w: float

    def __init__(self, node, cur_lvl, w, x_size, loss):
        self.attributes = []
        self.parents = []
        self.sum_error = 0
        self.size = 0
        self.s_upper = 0
        self.s_lower = 0
        self.score = 0
        self.error = 0
        self.max_tuple_error = 0
        self.x_size = x_size
        self.w = w
        self.loss = loss
        if cur_lvl == 0:
            self.key = node
            self.attributes.append(node)
        else:
            self.attributes = node.attributes
            self.key = node.attributes
        self.name = self.make_name()
        self.__hash__()

    def __hash__(self):
        return hash(self.name)

    def __add__(self, other):
        self.size += other.size
        self.sum_error += other.sum_error
        return self

    def combine_with(self, other):
        self.size = max(self.size, other.size)
        self.sum_error = max(self.sum_error, other.sum_error)
        return self

    def minimize_bounds(self, other):
        minimized = min(self.s_upper, other.s_upper)
        self.s_upper = minimized
        minimized = min(other.s_lower, self.s_lower)
        self.s_lower = minimized
        minimized = min(other.e_upper, self.e_upper)
        self.e_upper = minimized
        minimized = min(other.e_max_upper, self.e_max_upper)
        self.e_max_upper = minimized
        c_upper = self.calc_c_upper(self.w, self.x_size, self.loss)
        minimized = min(c_upper, self.c_upper)
        self.c_upper = minimized

    def __eq__(self, other):
        return (
            self.__hash__ == other.__hash__ and self.key == other.key)

    def update_metrics(self, row, loss_type):
        self.size += 1
        if loss_type == 0:
            self.sum_error += row[2]
            if row[2] > self.max_tuple_error:
                self.max_tuple_error = row[2]
        else:
            if row[2] != 0:
                self.sum_error += 1

    def calc_error(self):
        if self.size != 0:
            self.error = self.sum_error / self.size
        else:
            self.error = 0
        self.e_max = self.max_tuple_error
        self.e_max_upper = self.e_max
        self.e_upper = self.error

    def check_constraint(self, top_k, x_size, alpha):
        return self.score >= top_k.min_score and self.size >= x_size / alpha

    def make_name(self):
        name = ""
        for attribute in self.attributes:
            name = name + str(attribute) + " && "
        name = name[0: len(name) - 4]
        return name

    def calc_bounds(self, w, x_size, loss):
        self.s_upper = self.calc_s_upper()
        self.s_lower = self.calc_s_lower(x_size)
        self.e_upper = self.calc_e_upper()
        self.e_max_upper = self.calc_e_max_upper()
        self.c_upper = self.calc_c_upper(w, x_size, loss)

    def check_bounds(self, x_size, alpha, top_k):
        return self.s_upper >= x_size / alpha and self.c_upper >= top_k.min_score

    def calc_s_upper(self):
        cur_min = self.parents[0].size
        for parent in self.parents:
            cur_min = min(cur_min, parent.s_upper)
        return cur_min

    def calc_e_max_upper(self):
        e_max_min = self.parents[0].e_max_upper
        for parent in self.parents:
            e_max_min = min(e_max_min, parent.e_max_upper)
        return e_max_min

    def calc_s_lower(self, x_size):
        size_value = x_size
        for parent in self.parents:
            size_value = size_value - (size_value - parent.s_lower)
        return max(size_value, 1)

    def calc_e_upper(self):
        prev_e_uppers = []
        for parent in self.parents:
            prev_e_uppers.append(parent.e_upper)
        return float(min(prev_e_uppers))

    def calc_c_upper(self, w, x_size, loss):
        upper_score = w * (self.e_upper / self.s_lower) / (loss / x_size) + (
                    1 - w) * self.s_upper
        return float(upper_score)

    def print_debug(self, topk):
        print("new node has been created: " + self.make_name() + "\n")
        print("s_upper = " + str(self.s_upper))
        print("s_lower = " + str(self.s_lower))
        print("e_upper = " + str(self.e_upper))
        print("c_upper = " + str(self.c_upper))
        print("current topk min score = " + str(topk.min_score))
        print("-------------------------------------------------------------------------------------")

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

class Topk:
    k: int
    min_score: float
    slices: []
    keys: []

    def __init__(self, k):
        self.k = k
        self.slices = []
        self.keys = []
        self.min_score = 1

    def top_k_min_score(self):
        self.slices.sort(key=lambda x: x.score, reverse=True)
        self.min_score = self.slices[len(self.slices) - 1].score
        return self.min_score

    def add_new_top_slice(self, new_top_slice):
        self.min_score = new_top_slice.score
        if len(self.slices) < self.k:
            self.slices.append(new_top_slice)
            self.keys.append(new_top_slice.key)
            return self.top_k_min_score()
        else:
            self.slices[len(self.slices) - 1] = new_top_slice
            self.keys[len(self.slices) - 1] = new_top_slice.key
            return self.top_k_min_score()

    def print_topk(self):
        for candidate in self.slices:
            print(candidate.name + ": " + "score = " + str(candidate.score) + "; size = " + str(candidate.size))

    def buckets_top_k(self, cur_lvl_slices, x_size, alpha, cur_min):
        for bucket in cur_lvl_slices:
            if bucket.size > x_size / alpha and bucket.score >= cur_min:
                self.add_new_top_slice(bucket)
        return self

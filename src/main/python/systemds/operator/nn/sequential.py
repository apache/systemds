# -------------------------------------------------------------
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
# -------------------------------------------------------------
from systemds.operator.nn.layer import Layer


class Sequential(Layer):
    def __init__(self, *args):
        super().__init__()

        self.layers = []
        if len(args) == 1 and isinstance(args[0], list):
            self.layers = args[0]
        else:
            self.layers = list(args)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __setitem__(self, idx, value):
        self.layers[idx] = value

    def __delitem__(self, idx):
        del self.layers[idx]

    def __iter__(self):
        return iter(self.layers)

    def push(self, layer: Layer):
        """
        Add layer
        :param layer: Layer
        :return:
        """
        self.layers.append(layer)

    def pop(self):
        """
        Remove last layer
        :return: Layer
        """
        return self.layers.pop()

    def _instance_forward(self, X):
        """
        Forward pass
        :param X: Input matrix
        :return: output matrix
        """
        out = X
        for layer in self:
            out = layer.forward(out)
        return out

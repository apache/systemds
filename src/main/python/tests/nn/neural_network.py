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
from systemds.context import SystemDSContext

from systemds.operator.nn.affine import Affine
from systemds.operator.nn.relu import ReLU
from systemds.operator import Matrix, Source


class NeuralNetwork:
    _source: Source = None
    _X: Matrix

    def __init__(self, sds: SystemDSContext, dim: int):
        # first hidden layer
        self.affine1 = Affine(sds, dim, 128, seed=42)
        self.w1, self.b1 = self.affine1.weight, self.affine1.bias
        self.relu1 = ReLU(sds)

        # second hidden layer
        self.affine2 = Affine(sds, 128, 64, seed=42)
        self.w2, self.b2 = self.affine2.weight, self.affine2.bias
        self.relu2 = ReLU(sds)

        # third hidden layer
        self.affine3 = Affine(sds, 64, 32, seed=42)
        self.relu3 = ReLU(sds)
        self.w3, self.b3 = self.affine3.weight, self.affine3.bias

        # output layer
        self.affine4 = Affine(sds, 32, 2, seed=42)
        self.w4, self.b4 = self.affine4.weight, self.affine4.bias

    def forward_static_pass(self, X: Matrix) -> Matrix:
        """
        Compute forward pass through the network using static affine and relu calls
        :param X: Input matrix
        :return: Output matrix
        """
        X = self.affine1.forward(X)
        X = self.relu1.forward(X)

        X = self.affine2.forward(X)
        X = self.relu2.forward(X)

        X = self.affine3.forward(X)
        X = self.relu3.forward(X)

        X = self.affine4.forward(X)

        return X

    def forward_dynamic_pass(self, X: Matrix) -> Matrix:
        """
        Compute forward pass through the network using dynamic affine and relu calls
        :param X: Input matrix
        :return: Output matrix
        """
        X = Affine.forward(X, self.w1, self.b1)
        X = ReLU.forward(X)

        X = Affine.forward(X, self.w2, self.b2)
        X = ReLU.forward(X)

        X = Affine.forward(X, self.w3, self.b3)
        X = ReLU.forward(X)

        X = Affine.forward(X, self.w4, self.b4)

        return X

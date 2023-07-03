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
import os

from systemds.context import SystemDSContext
from systemds.operator import Matrix, Source, MultiReturn
from systemds.utils.helpers import get_path_to_script_layers


class Affine:
    _source: Source = None
    _X: Matrix
    weight: Matrix
    bias: Matrix

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, sds_context: SystemDSContext, d, m, seed=-1):
        """
        sds_context: systemdsContext
        d: number of features
        m: number of neuron
        """
        Affine._create_source(sds_context)

        # bypassing overload limitation in python
        self.forward = self._instance_forward
        self.backward = self._instance_backward

        # init weight and bias
        self.weight = Matrix(sds_context, '')
        self.bias = Matrix(sds_context, '')
        params_dict = {'D': d, 'M': m, 'seed': seed}
        out = [self.weight, self.bias]
        op = MultiReturn(sds_context, "affine::init", output_nodes=out, named_input_nodes=params_dict)
        self.weight._unnamed_input_nodes = [op]
        self.bias._unnamed_input_nodes = [op]
        op._source_node = self._source

    @staticmethod
    def forward(X: Matrix, W: Matrix, b: Matrix):
        """
        X: input matrix
        W: weigth matrix
        b: bias
        return out: output matrix
        """
        Affine._create_source(X.sds_context)
        out = Affine._source.forward(X, W, b)
        return out

    @staticmethod
    def backward(dout, X: Matrix, W: Matrix, b: Matrix):
        """
        dout: gradient of output, passed from the upstream
        X: input matrix
        W: weigth matrix
        b: bias
        return dX, dW,db: gradient of input, weights and bias, respectively
        """
        sds = X.sds_context
        Affine._create_source(sds)
        params_dict = {'dout': dout, 'X': X, 'W': W, 'b': b}
        dX = Matrix(sds, '')
        dW = Matrix(sds, '')
        db = Matrix(sds, '')
        out = [dX, dW, db]
        op = MultiReturn(sds, "affine::backward", output_nodes=out, named_input_nodes=params_dict)
        dX._unnamed_input_nodes = [op]
        dW._unnamed_input_nodes = [op]
        db._unnamed_input_nodes = [op]
        op._source_node = Affine._source
        return op

    def _instance_forward(self, X: Matrix):
        """
        X: input matrix
        return out: output matrix
        """
        self._X = X
        return Affine.forward(X, self.weight, self.bias)

    def _instance_backward(self, dout: Matrix):
        """
        dout: gradient of output, passed from the upstream
        return dX, dW,db: gradient of input, weights and bias, respectively
        """
        return Affine.backward(dout, self._X, self.weight, self.bias)

    @staticmethod
    def _create_source(sds: SystemDSContext):
        if Affine._source is None:
            path = get_path_to_script_layers()
            path = os.path.join(path, "affine.dml")
            Affine._source = sds.source(path, "affine")

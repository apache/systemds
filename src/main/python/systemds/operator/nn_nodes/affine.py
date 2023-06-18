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
from systemds.operator import Matrix, Source, MultiReturn
from systemds.utils.consts import PATH_TO_LAYERS_SCRIPT

PATH = PATH_TO_LAYERS_SCRIPT + "affine.dml"


class Affine:
    _sds_context: SystemDSContext
    _source: Source
    _X: Matrix
    weight: Matrix
    bias: Matrix

    def __init__(self, d, m, seed=-1):
        """
        d: number of features
        m: number of neuron
        """
        self._sds_context = SystemDSContext()
        self._source = self._sds_context.source(PATH, "affine")
        self.weight = Matrix(self._sds_context, '')
        self.bias = Matrix(self._sds_context, '')

        params_dict = {'D': d, 'M': m, 'seed': seed}
        out = [self.weight, self.bias]
        op = MultiReturn(self._sds_context, "affine::init", output_nodes=out, named_input_nodes=params_dict)
        self.weight._unnamed_input_nodes = [op]
        self.bias._unnamed_input_nodes = [op]
        op._source_node = self._source


    def forward(self, X):
        """
        X: input matrix
        return out: output matrix
        """
        self._X = X
        out = self._source.forward(X, self.weight, self.bias)
        return out

    def backward(self, dout):
        """
        dout: gradient of output, passed from the upstream
        return dX, dW,db: gradient of input, weights and bias, respectively
        """
        params_dict = {'dout': dout, 'X': self._X, 'W': self.weight, 'b': self.bias}
        dX = Matrix(self._sds_context, '')
        dW = Matrix(self._sds_context, '')
        db = Matrix(self._sds_context, '')
        out = [dX, dW, db]
        op = MultiReturn(self._sds_context, "affine::backward", output_nodes=out, named_input_nodes=params_dict)
        dX._unnamed_input_nodes = [op]
        dW._unnamed_input_nodes = [op]
        db._unnamed_input_nodes = [op]
        op._source_node = self._source
        return dX, dW, db

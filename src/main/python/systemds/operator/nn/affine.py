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
from systemds.operator import Matrix, MultiReturn
from systemds.operator.nn.layer import Layer


class Affine(Layer):
    weight: Matrix
    bias: Matrix

    def __init__(self, sds_context: SystemDSContext, d, m, seed=-1):
        """
        sds_context: The systemdsContext to construct the layer inside of
        d: The number of features that are input to the affine layer
        m: The number of neurons that are contained in the layer, 
            and the number of features output
        """
        super().__init__(sds_context, 'affine.dml')

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
        X: An input matrix
        W: The hidden weights for the affine layer
        b: The bias added in the output.
        return out: An output matrix.
        """
        Affine._create_source(X.sds_context, "affine.dml")
        return Affine._source.forward(X, W, b)

    @staticmethod
    def backward(dout:Matrix, X: Matrix, W: Matrix, b: Matrix):
        """
        dout: The gradient of the output, passed from the upstream
        X: The input matrix of this layer
        W: The hidden weights for the affine layer
        b: The bias added in the output
        return dX, dW, db: The gradients of: input X, weights and bias.
        """
        sds = X.sds_context
        Affine._create_source(sds, "affine.dml")
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
        X: The input matrix
        return out: The output matrix
        """
        self._X = X
        return Affine.forward(X, self.weight, self.bias)

    def _instance_backward(self, dout: Matrix, X: Matrix):
        """
        dout: The gradient of the output, passed from the upstream layer
        X: The input to this layer.
        return dX, dW,db: gradient of input, weights and bias, respectively
        """
        return Affine.backward(dout, X, self.weight, self.bias)

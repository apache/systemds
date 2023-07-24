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
import os.path

from systemds.context import SystemDSContext
from systemds.operator import Matrix, Source
from systemds.utils.helpers import get_path_to_script_layers


class ReLU:
    _source: Source = None
    _X: Matrix

    def __init__(self, sds: SystemDSContext):
        ReLU._create_source(sds)
        self.forward = self._instance_forward
        self.backward = self._instance_backward

    @staticmethod
    def forward(X: Matrix):
        """
        X: input matrix
        return out: output matrix
        """
        ReLU._create_source(X.sds_context)
        out = ReLU._source.forward(X)
        return out

    @staticmethod
    def backward(dout: Matrix, X: Matrix):
        """
        dout: gradient of output, passed from the upstream
        X: input matrix
        return dX: gradient of input
        """
        ReLU._create_source(dout.sds_context)
        dX = ReLU._source.backward(dout, X)
        return dX

    # forward = staticmethod(forward)
    # backward = staticmethod(backward)

    def _instance_forward(self, X: Matrix):
        self._X = X
        return ReLU.forward(X)

    def _instance_backward(self, dout: Matrix):
        return ReLU.backward(dout, self._X)

    @staticmethod
    def _create_source(sds: SystemDSContext):
        if ReLU._source is None:
            path = get_path_to_script_layers()
            path = os.path.join(path, "relu.dml")
            ReLU._source = sds.source(path, "relu")


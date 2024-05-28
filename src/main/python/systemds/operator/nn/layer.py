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
from systemds.operator import Source
from systemds.utils.helpers import get_path_to_script_layers


class Layer:
    """
    Interface for neural network layers
    """

    _source: Source = None

    def __init__(self, sds_context: SystemDSContext = None, dml_script: str = None):
        if sds_context is not None and dml_script is not None:
            self.__class__._create_source(sds_context, dml_script)

        # bypassing overload limitation in python
        self.forward = self._instance_forward
        self.backward = self._instance_backward

    @classmethod
    def _create_source(cls, sds_context: SystemDSContext, dml_script: str):
        """
        Create SystemDS source
        :param sds_context: SystemDS context
        :param dml_script: DML script inside /scripts/nn/layers/
        :return:
        """
        if cls._source is None or cls._source.sds_context != sds_context:
            script_path = get_path_to_script_layers()
            path = os.path.join(script_path, dml_script)
            name = dml_script.split(".")[0]
            cls._source = sds_context.source(path, name)

    def _instance_forward(self, *args):
        raise NotImplementedError

    def _instance_backward(self, *args):
        raise NotImplementedError

    @staticmethod
    def forward(*args):
        raise NotImplementedError

    @staticmethod
    def backward(*args):
        raise NotImplementedError

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
from systemds.scuro.utils.identifier import Identifier


class Representation:
    def __init__(self, name, parameters):
        self.name = name
        self._parameters = parameters
        self.self_contained = True
        self.representation_id = Identifier().new_id()

    @property
    def parameters(self):
        return self._parameters

    def get_current_parameters(self):
        current_params = {}
        if not self.parameters:
            return current_params

        for parameter in list(self.parameters.keys()):
            current_params[parameter] = getattr(self, parameter)
        return current_params

    def set_parameters(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

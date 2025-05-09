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
import abc
from typing import List

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.representation import Representation


class Context(Representation):
    def __init__(self, name, parameters = None):
        """
        Parent class for different context operations
        :param name: Name of the context operator
        """
        super().__init__(name) # TODO add parameters

    @abc.abstractmethod
    def execute(self, modality: Modality):
        """
        Implemented for every child class and creates a contextualized representation for a given modality
        :param modality: modality to use
        :return: contextualized data
        """
        raise f"Not implemented for Context Operator: {self.name}"

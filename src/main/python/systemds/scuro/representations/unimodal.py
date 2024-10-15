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
from systemds.scuro.representations.representation import Representation


class UnimodalRepresentation(Representation):
    def __init__(self, name):
        """
        Parent class for all unimodal representation types
        :param name: name of the representation
        """
        super().__init__(name)

    def parse_all(self, file_path, indices):
        raise f"Not implemented for {self.name}"


class PixelRepresentation(UnimodalRepresentation):
    def __init__(self):
        super().__init__("Pixel")

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
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.representations.unimodal import UnimodalRepresentation, HDF5, NPY, Pickle, JSON
from systemds.scuro.representations.lstm import LSTM


__all__ = ["Representation",
           "Average",
           "Concatenation",
           "Fusion",
           "UnimodalRepresentation",
           "HDF5",
           "NPY",
           "Pickle",
           "JSON",
           "LSTM"]

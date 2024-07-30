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
import json
import pickle

import h5py
import numpy as np

from representations.representation import Representation


class UnimodalRepresentation(Representation):
    def __init__(self, name):
        """
        Parent class for all unimodal representation types
        :param name: name of the representation
        """
        super().__init__(name)
    
    def parse_all(self, file_path, indices):
        raise f'Not implemented for {self.name}'


class PixelRepresentation(UnimodalRepresentation):
    def __init__(self):
        super().__init__('Pixel')


class ResNet(UnimodalRepresentation):
    def __init__(self):
        super().__init__('ResNet')


class Pickle(UnimodalRepresentation):
    def __init__(self):
        super().__init__('Pickle')
    
    def parse_all(self, filepath, indices):
        with open(filepath, "rb") as file:
            data = pickle.load(file, encoding='latin1')
        
        if indices is not None:
            for n, idx in enumerate(indices):
                result = np.empty((len(data), np.mean(data[idx][()], axis=1).shape[0]))
                break
            for n, idx in enumerate(indices):
                result[n] = np.mean(data[idx], axis=1)
            return result
        else:
            return np.array([np.mean(data[index], axis=1) for index in data])


class JSON(UnimodalRepresentation):
    def __init__(self):
        super().__init__('JSON')
    
    def parse_all(self, filepath, indices):
        with open(filepath) as file:
            return json.load(file)


class NPY(UnimodalRepresentation):
    def __init__(self):
        super().__init__('NPY')
    
    def parse_all(self, filepath, indices):
        data = np.load(filepath)
        
        if indices is not None:
            return np.array([data[n, 0] for n, index in enumerate(indices)])
        else:
            return np.array([data[index, 0] for index in data])


class HDF5(UnimodalRepresentation):
    def __init__(self):
        super().__init__('HDF5')
    
    def parse_all(self, filepath, indices=None):
        data = h5py.File(filepath)
        if indices is not None:
            return np.array([np.mean(data[index][()], axis=0) for index in indices])
        else:
            return np.array([np.mean(data[index][()], axis=0) for index in data])

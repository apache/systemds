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
from importlib.util import find_spec
from itertools import chain
from typing import Dict, Iterable

from systemds.utils.consts import MODULE_NAME


def create_params_string(unnamed_parameters: Iterable[str], named_parameters: Dict[str, str]) -> str:
    """
    Creates a string for providing parameters in dml. Basically converts both named and unnamed parameter
    to a format which can be used in a dml function call.

    :param unnamed_parameters: the unnamed parameter variables
    :param named_parameters: a dictionary of parameter names and variable names
    :return: the string to represent all parameters
    """
    named_input_strs = (f'{k}={v}' for (k, v) in named_parameters.items())
    return ','.join(chain(unnamed_parameters, named_input_strs))


def get_module_dir() -> os.PathLike:
    """
    Gives the path to our module

    :return: path to our module
    """
    spec = find_spec(MODULE_NAME)
    return spec.submodule_search_locations[0]


def get_slice_string(i):
    if isinstance(i, tuple):
        if len(i) > 2:
            raise ValueError(
                f'Invalid number of dimensions to slice {len(i)}, Only 2 dimensions allowed')
        else:
            return f'{get_slice_string(i[0])},{get_slice_string(i[1])}'
    elif isinstance(i, slice):
        if i.step:
            raise ValueError("Invalid to slice with step in systemds")
        elif i.start == None and i.stop == None:
            return ''
        elif i.start == None or i.stop == None:
            raise NotImplementedError("Not Implemented slice with dynamic end")
        else:
            # + 1 since R and systemDS is 1 indexed.
            return f'{i.start+1}:{i.stop}'
    else:
        # + 1 since R and systemDS is 1 indexed.
        sliceIns = i+1
    return sliceIns

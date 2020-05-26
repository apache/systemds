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

from typing import Dict, Iterable, Tuple

from systemds.context import SystemDSContext
from systemds.operator import OperationNode
from systemds.utils.consts import VALID_INPUT_TYPES


class Federated(OperationNode):

    def __init__(self, sds_context: SystemDSContext, addresses: Iterable[str],
                 ranges: Iterable[Tuple[Iterable[int], Iterable[int]]], *args,
                 **kwargs: Dict[str, VALID_INPUT_TYPES]) -> OperationNode:
        """Create federated matrix object.

        :param sds_context: the SystemDS context
        :param addresses: addresses of the federated workers
        :param ranges: for each federated worker a pair of begin and end index of their held matrix
        :param args: unnamed params
        :param kwargs: named params
        :return: the OperationNode representing this operation
        """
        addresses_str = 'list(' + \
            ','.join(map(lambda s: f'"{s}"', addresses)) + ')'
        ranges_str = 'list('
        for begin, end in ranges:
            ranges_str += f'list({",".join(map(str, begin))}), list({",".join(map(str, end))}),'
        ranges_str = ranges_str[:-1]
        ranges_str += ')'
        named_params = {'addresses': addresses_str, 'ranges': ranges_str}
        named_params.update(kwargs)
        super().__init__(sds_context, 'federated', args, named_params)

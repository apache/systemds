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
# Python
import logging
from systemds.context import SystemDSContext

# Create a federated matrix
# Indicate the dimensions of the data:
# Here the first list in the tuple is the top left Coordinate,
# and the second the bottom left coordinate.
# It is ordered as [col,row].
dims = ([0, 0], [3, 3])

# Specify the address + file path from worker:
address = "localhost:8001/temp/test.csv"

with SystemDSContext() as sds:
    fed_a = sds.federated([address], [dims])
    # Sum the federated matrix and call compute to execute
    logging.info(fed_a.sum().compute())
    # Result should be 45.

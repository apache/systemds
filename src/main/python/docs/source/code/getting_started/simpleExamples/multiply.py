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

import logging

from systemds.context import SystemDSContext

# Create a context and if necessary (no SystemDS py4j instance running)
# it starts a subprocess which does the execution in SystemDS
with SystemDSContext() as sds:
    # Full generates a matrix completely filled with one number.
    # Generate a 5x10 matrix filled with 4.2
    m = sds.full((5, 10), 4.20)
    # multiply with scalar. Nothing is executed yet!
    m_res = m * 3.1
    # Do the calculation in SystemDS by calling compute().
    # The returned value is an numpy array that can be directly printed.
    logging.info(m_res.compute())
    # context will automatically be closed and process stopped

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

import numpy as np
from systemds.context import SystemDSContext

# create a random array
m1 = np.array(np.random.randint(100, size=5 * 5) + 1.01, dtype=np.double)
m1.shape = (5, 5)
# create another random array
m2 = np.array(np.random.randint(5, size=5 * 5) + 1, dtype=np.double)
m2.shape = (5, 5)

# Create a context
with SystemDSContext() as sds:
    # element-wise matrix multiplication, note that nothing is executed yet!
    m_res = sds.from_numpy(m1) * sds.from_numpy(m2)
    # lets do the actual computation in SystemDS! The result is an numpy array
    m_res_np = m_res.compute()
    logging.info(m_res_np)

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
import numpy as np
from systemds.context import SystemDSContext

addr1 = "localhost:8001/temp/test.csv"
addr2 = "localhost:8002/temp/test.csv"
addr3 = "localhost:8003/temp/test.csv"

# Create a federated matrix using two federated environments
# Note that the two federated matrices are stacked on top of each other

with SystemDSContext() as sds:

    fed_a = sds.federated([addr1],[([0, 0], [3, 3])])
    fed_b = sds.federated([addr2],[([0, 0], [3, 3])])
    # fed_c = sds.federated([addr3],[([0, 0], [3, 3])])

    np_array = np.array([[1,2,3],[4,5,6],[7,8,9]])

    loc_a = sds.from_numpy(np_array)
    loc_b = sds.from_numpy(np_array)

    fed_res = fed_a @ fed_b
    loc_res = loc_a @ loc_b

    hybrid_res_1 = fed_a @ loc_b
    hybrid_res_2 = loc_a @ fed_b

    # compute and print
    print(fed_a.compute())
    print(fed_b.compute())
    print(fed_res.compute(verbose=True))
    print(loc_res.compute(verbose=True))
    print(hybrid_res_1.compute())
    print(hybrid_res_1.compute())

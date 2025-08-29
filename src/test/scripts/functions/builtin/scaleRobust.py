#-------------------------------------------------------------
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
#-------------------------------------------------------------

import sys
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix
from sklearn.preprocessing import RobustScaler

if __name__ == "__main__":
    input_path = sys.argv[1] + "A.mtx"
    output_path = sys.argv[2] + "B"

    X = mmread(input_path).toarray()

    # Apply RobustScaler
    scaler = RobustScaler()
    Y = scaler.fit_transform(X)

    mmwrite(output_path, csc_matrix(Y))

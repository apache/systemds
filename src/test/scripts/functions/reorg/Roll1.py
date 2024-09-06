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
from scipy.sparse import csr_matrix

# Read command line arguments
input_file = sys.argv[1] + "A.mtx"
output_file = sys.argv[2] + "B"

# Read the matrix from a Matrix Market file
A = mmread(input_file).toarray()  # Read as a dense array

# Reverse each column
B = np.roll(A, axis=None, shift=1)

# Convert back to sparse format for saving
B_sparse = csr_matrix(B)

# Write the modified matrix to the output file in Matrix Market format
mmwrite(output_file, B_sparse, precision=22, symmetry='general')


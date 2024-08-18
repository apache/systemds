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

import os
import numpy as np
import pandas as pd

def gen_data_sparse(rows, cols, density, path, chunk_size=10000):
    """
    Generate a sparse matrix with given density and save it to a CSV file in a dense format.

    Parameters:
    rows (int): Number of rows.
    cols (int): Number of columns.
    density (float): Fraction of non-zero elements.
    path (str): Path to save the generated matrix.
    chunk_size (int): Number of rows per chunk to generate and save.
    """
    with open(path, 'w') as f:
        for start_row in range(0, rows, chunk_size):
            end_row = min(start_row + chunk_size, rows)
            chunk_rows = end_row - start_row

            chunk_matrix = np.zeros((chunk_rows, cols))

            n_nonzero = int(density * chunk_rows * cols)
            nonzero_indices = (np.random.randint(chunk_rows, size=n_nonzero), np.random.randint(cols, size=n_nonzero))
            chunk_matrix[nonzero_indices] = np.random.random(n_nonzero)

            np.savetxt(f, chunk_matrix, delimiter=',')
            #np.savetxt(f, chunk_matrix, delimiter=',', fmt='%.10f')

            print(f"Saved chunk {start_row} to {end_row} to {path}")

def main():
    # Hardcoded parameters
    sparse_gb = 0.0001
    sparsity_values = [0.0001, 0.001, 0.01, 0.1]

    current_directory = os.getcwd()
    target_directory = os.path.abspath(os.path.join(current_directory, '../../../../src/test/resources/datasets/slab/sparse'))
    os.makedirs(target_directory, exist_ok=True)


    for sr in sparsity_values:
        stub = str(sr).replace('.', '_')
        stub = "sparsity_"+stub
        k = int(np.ceil((sparse_gb * 1e9) / float(8 * 100)))

        # Paths for saving the matrices
        mpath_tall = os.path.join(target_directory, f'M_{stub}_tall.csv')
        mpath_wide = os.path.join(target_directory, f'M_{stub}_wide.csv')

        # Generate and save sparse matrices
        gen_data_sparse(k, 100, sr, mpath_tall)
        gen_data_sparse(100, k, sr, mpath_wide)

if __name__ == "__main__":
    main()

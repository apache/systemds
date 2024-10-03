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


from systemds.operator import Matrix, Scalar


def solve(a: Matrix, b: Matrix) -> "Matrix":
    """
     Computes the least squares solution for system of linear equations A %*% x = b i.e., it finds x such that
     ||A%*%x – b|| is minimized. The solution vector x is computed using a QR decomposition of A.

    :param a: (m,n) matrix a
    :param b: (m,1) matrix b
    :return: (n, 1) matrix x
    """

    return Matrix(a.sds_context, "solve", [a, b])

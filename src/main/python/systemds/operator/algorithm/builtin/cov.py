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


def cov(x: Matrix, y: Matrix, w: Matrix = None) -> "Scalar":
    """
     Returns the covariance between two 1-dimensional column matrices X and Y. The function takes an optional weights
     parameter W. All column matrices X, Y, and W (when specified) must have the exact same dimension.

    :param x: 1-dimensional column matrices x
    :param y: 1-dimensional column matrices y
    :param w: (optional)  1-dimensional weight column matrix
    :return: Covariance of the input matrices
    """
    if w is None:
        return Scalar(x.sds_context, "cov", [x, y])
    else:
        return Scalar(x.sds_context, "cov", [x, y, w])

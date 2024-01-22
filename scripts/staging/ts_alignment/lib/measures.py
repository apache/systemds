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

import numpy as np
from lib.distance_measure import Measure
from typing import List


class Measures:
    def __init__(self, x, y, measures: List[Measure], cross_correlation=False, get_diagonals=False, get_average=True):
        self.x = x
        self.y = y

        try:
            if self.x.shape[1]:
                self.is_multivariate = True
                assert self.x.shape[1] == self.y.shape[
                    1], f'The number of components in x {self.x.shape[1]} does ' \
                        f'not match the number of components in y {self.y.shape[1]}! '
        except:
            self.is_multivariate = False

        self.m = self.x.shape[0]
        self.n = self.y.shape[0]

        if len(measures) == 0:
            raise 'No distance measure provided!'

        self.measures = measures
        self.cross_correlation = cross_correlation
        self.get_diagonals = get_diagonals
        self.get_average = get_average
        self.constant = -999

        if self.cross_correlation and not (self.get_diagonals or self.get_average):
            raise 'To get a valid cross correlation please set \'get_diagonals\' or \'get_average\' to True'

    def compute(self):
        for measure in self.measures:
            if self.is_multivariate:
                dist_matrix = measure.compute_mv_distances(self.x, self.y)
                if self.cross_correlation:
                    if self.get_average:
                        measure.result = self._compute_average_from_diagonal_matrix(dist_matrix)
                    else:
                        measure.result = self._compute_diagonals(dist_matrix)
                else:
                    measure.result = np.mean(np.diagonal(dist_matrix))

                if measure.file_writer is not None:
                    measure.file_writer.writerow(measure.result)
            else:
                dist_matrix = self._compute_uv_measure(measure.uv_distance_fct)

                if measure.file_writer is not None:
                    measure.file_writer.writerow(dist_matrix)

                measure.result = dist_matrix.reshape(-1, 1)

    def _compute_uv_measure(self, distance_function):
        if not self.cross_correlation:
            return distance_function(self.x, self.y)

        result = np.zeros(self.m)
        x = self.x
        for k in range(0, self.m):
            if self.m + k > self.n:
                y_l = self.y[k:]
                x = self.x[0:len(y_l)]
            else:
                y_l = self.y[k:len(x) + k]

            result[k] = distance_function(x, y_l)
        return result

    def _compute_average_from_diagonal_matrix(self, matrix):
        average_per_lag = np.zeros(self.m)

        for lag in range(self.m):
            diagonal_p = np.diagonal(matrix, offset=lag)
            average_per_lag[lag] = (np.mean(diagonal_p))

        return average_per_lag

    def _compute_diagonals_for_unequal_rows(self, matrix):
        diagonals = np.empty((self.m, self.m))
        diagonals[:] = self.constant

        for lag in range(self.m):
            diag = (np.diagonal(matrix, offset=lag))

            if diag.shape[0] != self.m:
                diagonals[lag][:diag.shape[0]] = diag
            else:
                diagonals[lag] = diag

        return diagonals

    def _compute_diagonals(self, matrix):
        if 2 * self.m != self.n:
            return self._compute_diagonals_for_unequal_rows(matrix)

        diagonals = np.empty((self.m, self.m))
        diagonals[:] = self.constant

        for lag in range(self.m):
            diagonals[lag] = np.diagonal(matrix, offset=lag)

        return diagonals

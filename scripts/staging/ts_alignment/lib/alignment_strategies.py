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

import csv
import numpy as np
import pandas as pd
import os
import time
import warnings

from abc import ABC, abstractmethod

from pandas import DataFrame

from lib.data_loader import DataLoader
from datetime import timedelta
from fastdtw import fastdtw
from lib.measures import Measures
from lib.distance_measure import Measure
from scipy.spatial.distance import euclidean
from typing import Optional, List

from lib.sensor_data_loader import SensorDataLoader

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


class AlignmentStrategy(ABC):
    def __init__(self, x: DataLoader, y: DataLoader, folder_path=None, name=None, verbose=False):
        self.x = x
        self.y = y
        self.folder_path = folder_path
        if folder_path is not None:
            self.folder_path = f'{folder_path}/'
            self.create_result_folder(self.folder_path)
        self.verbose = verbose
        self.strategy_name = name

    @abstractmethod
    def execute(self) -> Optional[timedelta]:
        pass

    def create_result_folder(self, folder_name):
        if folder_name is None:
            raise 'Please provide a folder name for the results!'

        if not os.path.exists(f'{folder_name}/'):
            os.makedirs(f'{folder_name}/')


class Alignment:
    def __init__(self, strategy: AlignmentStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: AlignmentStrategy) -> None:
        if strategy is None:
            raise 'Please choose an alignment Strategy!'
        self._strategy = strategy

    def get_strategy(self):
        return self._strategy

    def compute_alignment(self) -> Optional[timedelta]:
        return self._strategy.execute()


class CCRAlignment(AlignmentStrategy):
    """
    CCR Alignment computes the cross correlation over the whole dataset and returns the time lag where the best
    similarity measure is detected.

    Params:
        x: DataLoader for dataset x
        y: DataLoader for dataset y
        measure: instance of measure to be used for cross correlation computation one of:
            [PearsonCorrelation() EuclideanDistance(), CosineSimilarity()]
        folder_name (Optional): folder name to store the results
        batch_size: integer value indicating the number of batches in which the dataset is processed

    """

    def __init__(self: str, x: DataLoader, y: DataLoader, measure: Measure,
                 folder_name=None, verbose=False):
        super().__init__(x, y, folder_path=folder_name, name='CCR Alignment', verbose=verbose)
        self._len_x = x.all_frames
        self._measure = measure
        self._constant = -999

    def compute_time_lags(self, start, num_batches) -> np.ndarray:
        lags = np.array([])
        if self.verbose:
            print(f'Start: {start} num_batches {num_batches}')

        num_batches_j = max(self.x.get_num_total_batches(), 1)

        if self.verbose:
            print(f'Executing {num_batches} batches')
            overall_start_time = time.time()

        if isinstance(self.x, SensorDataLoader) and self.x.column != -1:
            m = 1
        else:
            m = self.x.get_batch_size()

        for i in range(start, num_batches):
            self.x.reset()
            self.y.reset(frame_position=(i * self.y.get_batch_size()))
            b = np.ones(((max(self.x.get_num_total_batches(), 1) - i) * m, self.x.get_batch_size()), dtype=np.float32)
            b[:] = self._constant
            start_time = time.time()
            y_1 = self.y.extract()
            if self.verbose:
                print(f'{num_batches_j - i}')

            for j in range(0, num_batches_j - i):
                y_2 = self.y.extract()

                if y_1.shape[0] == 0:
                    break
                elif y_2.shape[0] == 0:
                    y_b = y_1
                else:
                    y_b = np.concatenate((y_1, y_2))

                x_b = self.x.extract()

                Measures(x_b, y_b, [self._measure], cross_correlation=True, get_diagonals=True,
                         get_average=False).compute()

                diagonal = self._measure.result.T

                b[j * m: (j + 1) * m][:diagonal.shape[0], :diagonal.shape[1]] = diagonal
                y_1 = y_2

            constant_mask = np.equal(b, self._constant)
            b[constant_mask] = np.nan

            lags = np.hstack((lags, np.nanmean(b, axis=0)))

            if self.verbose:
                print(f'{i}: {time.time() - start_time}')

        if self.folder_path is not None:
            file = open(f'{self.folder_path}/static_alignment_{self._measure.short_name}.csv', 'w')
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            writer.writerow(lags)

        if self.verbose:
            print(f'Overall execution took {time.time() - overall_start_time}')

        return lags

    def execute(self) -> timedelta:
        num_batches = max(self.x.get_num_total_batches() // 2, 1)

        lags = self.compute_time_lags(0, num_batches)

        if self.folder_path is not None:
            np.savetxt(f'{self.folder_path}/static_alignment.csv', lags, delimiter=",")

        max_index = np.argmax(lags)
        if max_index > self.x.time_stamps.shape[0]:
            max_index = self.x.time_stamps.shape[0] - 1
        max_value = np.max(lags)

        if self.verbose:
            print(f'Max index {max_index}')
            print(f'max value: {max_value}')

        time_lag = self.x.time_stamps.iloc[max_index] - self.x.time_stamps.iloc[0]

        if self.verbose:
            print(
                f'Max Correlation {max_value} at index {max_index} leads to a time lag of {time_lag.total_seconds()} seconds')

        return time_lag.total_seconds()


class ChunkedCCRAlignment(AlignmentStrategy):
    """
    Chunked CCR Alignment computes the cross correlation over chunks of the dataset and returns the time lag where the best
    similarity measure is detected within each chunk.

    Params:
        x: DataLoader for dataset x
        y: DataLoader for dataset y
        measures: list of measures to be used for cross correlation computation
            [PearsonCorrelation() EuclideanDistance(), CosineSimilarity()]
        folder_name: folder name where the results of the computations should be saved
        window_size: integer holding the window size in seconds
    """

    def __init__(self, x: DataLoader, y: DataLoader, measures: List[Measure], folder_name: str,
                 window_size: int, verbose=False, compute_uv=False):
        super().__init__(x, y, name='Chunked CCR Alignment', verbose=verbose)
        self._window_size = window_size
        self.folder_name = folder_name
        self.folder_path = self.folder_name

        self.measures = measures
        self.compute_uv = compute_uv

    def execute(self):

        if self.compute_uv and isinstance(self.x, SensorDataLoader):
            chunked_lags = []
            num_columns = self.x.df.shape[1]
            for i in range(0, num_columns):
                self.folder_path = self.folder_name + f'/{str(i)}'
                self.create_result_folder(self.folder_path)
                self.x.set_column(i)
                self.y.set_column(i)
                self.x.reset(self._window_size)
                self.y.reset(self._window_size)
                chunked_lags.append(self.compute_time_lags())
            self.x.df = self.x.all_columns
            self.y.df = self.y.all_columns
            return chunked_lags
        else:
            self.x.reset(self._window_size)
            self.y.reset(self._window_size)
            self.create_result_folder(self.folder_path)
            return self.compute_time_lags()

    def compute_time_lags(self) -> DataFrame:
        start = time.time()
        columns = [measure.name for measure in self.measures]
        columns.append('chunk')
        columns.append('offset')
        chunked_ccr_x_y_lag = pd.DataFrame(columns=columns)

        y_1 = self.y.extract()
        if self.verbose:
            print(f'Executing {self.x.get_num_total_batches()} batches...')

        for i in range(self.x.get_num_total_batches()):
            start_read = time.time()
            x = self.x.extract()
            y_2 = self.y.extract()

            if y_1.shape[0] == 0:
                break
            elif y_2.shape[0] == 0:
                y = y_1
            else:
                y = np.concatenate((y_1, y_2))
            if self.verbose:
                print(
                    f'{i}: reading data took {time.time() - start_read} seconds, size of x: {x.nbytes}, size of '
                    f'y: {y.nbytes}')

            if 0 < x.shape[0] and 0 < y.shape[0]:
                if y.shape[0] // 2 < x.shape[0]:
                    x = x[:y.shape[0] // 2]

                start_corr = time.time()
                Measures(x, y, self.measures, cross_correlation=True, get_average=True).compute()

                for measure in self.measures:
                    nan_mask = np.isnan(measure.result)

                    if measure.api_name != 'euclidean_distance':
                        measure.result[nan_mask] = 0.0
                        offset = np.argmax(measure.result) / self.x.fps
                        distance = np.max(measure.result)
                    else:
                        measure.result[nan_mask] = 1.0
                        offset = np.argmin(measure.result) / self.x.fps
                        distance = np.min(measure.result)

                    if len(chunked_ccr_x_y_lag) > 0 and chunked_ccr_x_y_lag['offset'].iloc[-1] == offset and \
                            chunked_ccr_x_y_lag['chunk'].iloc[-1] == i:
                        chunked_ccr_x_y_lag.loc[i, measure.name] = distance
                    else:
                        chunked_ccr_x_y_lag = pd.concat(df.dropna(axis=1, how='all') for df in [chunked_ccr_x_y_lag, pd.DataFrame(data={'chunk': i, 'offset': offset, measure.name: distance}, index=[0])]).reset_index(drop=True)

                if self.verbose:
                    print(
                        f'Calculation of cross correlations took {time.time() - start_corr:.2f} seconds')
                y_1 = y_2

        end = time.time()
        if self.verbose:
            print(f'Computation for {self._window_size} took {end - start}')

        out_path = f'{self.folder_path}/corr_for_{self._window_size}.csv'
        chunked_ccr_x_y_lag.to_csv(out_path)

        return chunked_ccr_x_y_lag


class DynamicTimeWarpingAlignment(AlignmentStrategy):
    """
    DTW Alignment using the fastdtw library function computes the dtw distance and the best alignment path between
    two provided time series

    Params:
        x: pointer to datastructure for dataset x
        y: pointer to datastructure for dataset y
    Returns:
        dynamic alignment path between x and y
    """

    def __init__(self, x, y, verbose=False):
        super().__init__(x, y, name='Dynamic Time Warping', verbose=verbose)

    def execute(self) -> List:
        distance, path = fastdtw(self.x.extract(), self.y.extract(), dist=euclidean)
        if self.verbose:
            print(f'DTW distance: {distance}')

        return path

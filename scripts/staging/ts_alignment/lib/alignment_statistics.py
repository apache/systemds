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

import pandas as pd
from lib.distance_measure import Measure
from typing import List


class AlignmentStatistics:
    def __init__(self, measures: List[Measure], index_name='Chunk Size') -> None:
        self.measures = measures
        self.index_name = index_name
        self.stats = {
            self.index_name: []
        }

        for measure in measures:
            self.stats.update({measure.name: []})

        self.max_score = 0
        self.best_measure_name = ''
        self.best_index = 0
        self.best_lags = None

        self.details = {self.index_name: [], 'Avg Offset': {}, 'Min Offset': {}, 'Max Offset': {}, 'lags': {}}

        for measure in measures:
            self.details['Avg Offset'].update({measure.name: []})
            self.details['Min Offset'].update({measure.name: []})
            self.details['Max Offset'].update({measure.name: []})
            self.details['lags'].update({measure.name: []})

    def get_df(self):
        df = pd.DataFrame.from_dict(self.stats)
        df = df.set_index([self.index_name])

    def add_index(self, index: float):
        self.details[self.index_name].append(index)
        self._add_value_for_key(self.index_name, index)

    def add_score_for_measure(self, measure: Measure, score: float):
        self._add_value_for_key(measure.name, score)

    def add_agg_score(self, score: float):
        self._add_value_for_key('Max Aggregate', score)

    def extract_stats(self, df, measure_name):
        if self.details.get('Avg Offset').get(measure_name) is None:
            self.details['Avg Offset'].update({measure_name: []})
            self.details['Min Offset'].update({measure_name: []})
            self.details['Max Offset'].update({measure_name: []})
            self.details['lags'].update({measure_name: []})

        self.details['Avg Offset'][measure_name].append(df.loc[:, 'offset'].mean())
        self.details['Min Offset'][measure_name].append(df.loc[:, 'offset'].min())
        self.details['Max Offset'][measure_name].append(df.loc[:, 'offset'].max())
        self.details['lags'][measure_name].append(df)

    def evaluate_best_stats(self):
        if self.stats.get('Max Aggregate') is not None:
            self._extract_stats_for_key('Max Aggregate')

        for measure in self.measures:
            self._extract_stats_for_key(measure.name)

    def print_best_score(self):
        self.evaluate_best_stats()

        print(
            f'Best score: {self.max_score} for measure: {self.best_measure_name} at {self.index_name}: {self.best_index}')

    def _add_value_for_key(self, key, value):
        if self.stats.get(key) is None:
            self.stats.update({key: []})
        self.stats[key].append(value)

    def _extract_stats_for_key(self, key):
        scores = self.stats[key]
        if max(scores) >= self.max_score:
            self.max_score = max(scores)
            self.best_measure_name = key
            idx = scores.index(max(scores))
            indices = self.stats[self.index_name]
            self.best_index = indices[idx]
            self.best_lags = self.details['lags'][key][idx]

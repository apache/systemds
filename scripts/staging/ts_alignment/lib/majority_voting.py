import pandas as pd
import numpy as np
import statistics

from lib.distance_measure import Measure
from typing import List


class MajorityVoting:
    """
    Majority Voting algorithm to find the best timelag between:
        1. multivariate time series
        2. different similarity measures

    Params:
        input_file_path: path to file that contains the time lags as computed in the ChunkedCCRAlignment alignment method
        num_cols:       -1 to compute the majority voting over all columns using different similarity measures,
                        number of columns in the input data to compare univariate time lags (and optionally different similarity measures)
        measures:       list of similarity measures to be compared [PearsonCorrelation() EuclideanDistance(), CosineSimilarity()]
        chunk_size:     the chunk size used in the alignment process
    """

    def __init__(self, input_file_path, num_cols, measures: List[Measure], chunk_size):
        self.measures = measures.copy()
        self.input_file_path = input_file_path
        self.num_cols = num_cols
        self.chunk_size = chunk_size
        self.dfs = []

    def calculate_majority_lags_for_measure(self, measure):
        self._parse_time_lags(measure)
        return self.calculate_majority_lags(aggregate_measures=False)

    def calculate_majority_lags(self, aggregate_measures=True):
        if len(self.dfs) == 0:
            self._parse_time_lags()

        df_len = [np.max(df.chunk) for df in self.dfs]
        num_chunks = np.max(df_len)
        df_majority = pd.DataFrame(columns=['offset'])
        for chunk in range(0, int(num_chunks) + 1):
            offsets = []
            for df in self.dfs:
                measures_for_current_chunk = df.loc[df['chunk'] == chunk]
                if len(measures_for_current_chunk['offset'].values) != 0:
                    if aggregate_measures:
                        offsets.append(self._aggregate_measures(measures_for_current_chunk))
                    else:
                        offsets.append(df.loc[df['chunk'] == chunk, 'offset'].values[0])
            sorted_offsets = sorted(offsets)
            median = statistics.median(sorted_offsets)
            df_majority.loc[chunk, 'offset'] = median

        return df_majority

    def _parse_dynamic_lags(self, path: str, measure: Measure):
        if measure is not None and measure in self.measures:
            self.measures.remove(measure)

        df = pd.read_csv(path, parse_dates=True)

        if measure is not None:
            df = df[~df[measure.name].isna()]
            measure_names = [measure.name for measure in self.measures]
            return df.drop(columns=measure_names)

        return df

    def _parse_time_lags(self, measure=None):
        if self.num_cols == -1:
            self.dfs.append(
                self._parse_dynamic_lags(
                    self.input_file_path + '/' + f'corr_for_{self.chunk_size}.csv',
                    measure))
        else:
            for i in range(0, self.num_cols):
                self.dfs.append(
                    self._parse_dynamic_lags(self.input_file_path + '/' + str(
                        i) + '/' + f'corr_for_{self.chunk_size}.csv', measure))

    def _aggregate_measures(self, measures):
        max_m_f = -1.0
        max_measure = ''
        for measure in self.measures:
            m = measures[measure.name].values
            m = m[~np.isnan(m)]
            if len(m) > 0:
                m = m[0]
            else:
                m = 0.0

            if measure.name == 'Euclidean Distance':
                if max_m_f > m:
                    max_measure = measure.name
                    max_m_f = m
            else:
                if max_m_f < m:
                    max_measure = measure.name
                    max_m_f = m

        offset = measures[~measures[max_measure].isna()].offset.values
        return offset[0] if len(offset) > 0 else 0.0

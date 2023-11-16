import abc
import pandas as pd

from abc import ABC


class DataLoader(ABC):
    def __init__(self, batch_size, x):
        self.all_frames = None
        self.df = None
        self.end_time = None
        self.fps = None
        self.time_stamps = None
        self.x = x

        self._aligned_start_time = None
        self._batch_size = batch_size
        self._start_time = None

    @abc.abstractmethod
    def set_properties(self):
        raise 'Method \'set_properties\' not implemented for this DataLoader!'

    @abc.abstractmethod
    def reset(self, batch_size_in_seconds: int, frame_position: int):
        raise 'Method \'reset\' not implemented for this DataLoader!'

    @abc.abstractmethod
    def extract(self, start_index=None, end_index=None):
        raise 'Method \'extract\' not implemented for this DataLoader!'

    def get_batch_size(self):
        return self._batch_size

    def set_batch_size(self, b_size):
        self._batch_size = b_size

    def get_start_time(self):
        return self._start_time

    def get_num_total_batches(self):
        return round(int((self.end_time - self._aligned_start_time).total_seconds() * self.fps) / self._batch_size)

    def set_aligned_start_time(self):
        if self._start_time is None:
            raise 'Start time not set!'

        if self.x is None:
            self._aligned_start_time = self._start_time
        else:
            self._aligned_start_time = self.x.get_start_time() if self._start_time < self.x.get_start_time() else self._start_time

    def get_aligned_start_time(self):
        return self._aligned_start_time

    def get_time_at_position(self, position):
        if position == -1:
            position = self.all_frames
        return pd.Timedelta(seconds=(position / self.fps))

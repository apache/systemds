import pandas as pd

from lib.data_loader import DataLoader


class SensorDataLoader(DataLoader):
    def __init__(self, path, batch_size, column=-1, x=None, fps=1000):
        super().__init__(batch_size, x)
        if path == '':
            raise 'Please provide a path for the csv file!'

        self.all_columns = None
        self.chunk = 0
        self.column = column
        self.fps = fps

        self.load_datasets(path)
        self.set_properties()

    def set_column(self, column_index):
        if self.all_columns is None:
            raise 'Dataframe can not be None!'

        self.column = column_index
        self.df = pd.DataFrame(self.all_columns.iloc[:, column_index], index=self.all_columns.index)

    def set_properties(self):
        self.all_frames = self.df.shape[0]
        self.end_time = self._start_time + pd.Timedelta(seconds=(self.all_frames / self.fps))
        self.time_stamps = pd.concat([self.time_stamps, pd.Series(self.df.index)], axis=0)
        self.set_batch_size(self.get_batch_size() if self.get_batch_size() is not None else self.all_frames)

    def reset(self, batch_size_in_seconds=None, frame_position=0, num_chunks=None):
        if batch_size_in_seconds is not None:
            self.set_batch_size(round(batch_size_in_seconds * self.fps))

        self.chunk = frame_position // self.get_batch_size()

    def extract(self, start_index=None, end_index=None):
        if self.get_batch_size() is None:
            return self.df.to_numpy()

        if start_index is not None:
            start = start_index
        else:
            start = int(self.chunk * self.get_batch_size())
        self.chunk += 1

        if self.df.shape[1] == 1:
            return self.df.iloc[start: start + self.get_batch_size()].to_numpy().reshape(-1)
        return self.df.iloc[start: start + self.get_batch_size()].to_numpy()

    def load_datasets(self, path):
        self.df = pd.read_csv(path, parse_dates=True, header=0, index_col=0, dtype=float)
        self.df.index = pd.to_datetime(self.df.index, format='%Y-%m-%d  %H:%M:%S.%f')
        self._start_time = self.df.index[0]
        self.end_time = self.df.index[-1].time()
        self.set_aligned_start_time()
        self.df = self.df.between_time(self._aligned_start_time.time(), self.end_time)
        self.df = self.df.sort_index()

        if self.x is not None:
            if self.x.df.shape[0] > self.df.shape[0]:
                self.x.all_columns = self.x.all_columns.iloc[:self.df.shape[0]]
                self.x.df = self.x.all_columns
            elif self.df.shape[0] > self.x.df.shape[0]:
                self.df = self.df.iloc[:self.x.df.shape[0]]

        for column_index in range(self.df.shape[1]):
            scaled = (self.df.iloc[:, column_index] - self.df.iloc[:, column_index].min()) / (
                    self.df.iloc[:, column_index].max() - self.df.iloc[:, column_index].min())

            self.df.iloc[:, column_index] = scaled

        self.all_columns = self.df

    def set_end_time(self, other_end_time):
        if self.end_time > other_end_time:
            self.end_time = other_end_time
        self.all_frames = int((self.end_time - self._aligned_start_time).total_seconds() * self.fps)
        self.df = self.df.iloc[:self.all_frames]
        self.all_columns = self.all_columns.iloc[:self.all_frames]

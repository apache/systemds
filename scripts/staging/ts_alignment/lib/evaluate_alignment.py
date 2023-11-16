import numpy as np

from lib.data_loader import DataLoader


class Evaluator:
    def __init__(self, x, y, data_loader: DataLoader):
        self.x = x
        self.y = y
        self.data_loader = data_loader

    def parse_lags(self, lags):
        if isinstance(lags, float):
            return self._parse_static_lags(lags)

        elements_per_chunk = self.data_loader.get_batch_size()
        lag_indices = np.ones((int(len(self.data_loader.df)), 2), dtype=int)
        lag_indices[:, :] = -1

        for index, row in lags.iterrows():
            start_c = int(index * elements_per_chunk)
            lag = round(row.offset * self.data_loader.fps)

            for i in range(start_c, start_c + elements_per_chunk * 2):
                if i + lag >= lag_indices.shape[0]:
                    break

                lag_indices[i, :] = [i, i + lag]

            start_c += elements_per_chunk

        lag_indices = lag_indices[~(lag_indices[:, 0:1] == -1).any(1)]

        return lag_indices

    def evaluate(self, lag_indices, univariate=True):
        if lag_indices is not None:
            aligned = np.zeros((lag_indices.shape[0], self.x.shape[1] + self.y.shape[1]))
            aligned[:, :self.x.shape[1]] = self.x.iloc[lag_indices[:, 0]]
            aligned[:, self.x.shape[1]:] = self.y.iloc[lag_indices[:, 1]]
        else:
            aligned = np.zeros((self.x.shape[0], self.x.shape[1] + self.y.shape[1]))
            aligned[:, :self.x.shape[1]] = self.x.to_numpy()
            aligned[:, self.x.shape[1]:] = self.y.iloc[:len(self.x)].to_numpy()

            aligned = aligned[~np.isnan(aligned).any(axis=1)]

        if univariate:
            average = []
            for i in range(0, self.x.shape[1]):
                corr = np.corrcoef(aligned[:, i], aligned[:, self.x.shape[1] + i])[0][1]
                average.append(corr)
        else:
            average = [np.corrcoef(aligned[i, :self.x.shape[1]], aligned[i, self.x.shape[1]:])[0][1] for i in
                       range(0, aligned.shape[0])]

        correlation = np.mean(average)
        print(f'Pearson correlation after alignment: {correlation}')
        return correlation

    def _parse_static_lags(self, lag):
        time_lag = int(lag * self.data_loader.fps)
        len_lags = len(self.x) - time_lag
        lags = np.ones((len_lags, 2), dtype=int)
        lags[:, :] = -1

        for i in range(0, len_lags):
            lags[i, :] = [i, i + time_lag]
        lags = lags[~(lags[:, 0:1] == -1).any(1)]
        return lags

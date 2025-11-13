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
import numpy as np
from typing import List, Optional, Union
import h5py


from systemds.scuro.dataloader.base_loader import BaseLoader
from systemds.scuro.modality.type import ModalityType


class TimeseriesLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        signal_names: List[str],
        data_type: Union[np.dtype, str] = np.float32,
        chunk_size: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        normalize: bool = True,
        file_format: str = "npy",
    ):
        super().__init__(
            source_path, indices, data_type, chunk_size, ModalityType.TIMESERIES
        )
        self.signal_names = signal_names
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.file_format = file_format.lower()

        if self.file_format not in ["npy", "mat", "hdf5", "txt"]:
            raise ValueError(f"Unsupported file format: {self.file_format}")

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        self.file_sanity_check(file)

        if self.file_format == "npy":
            data = self._load_npy(file)
        elif self.file_format in ["txt", "csv"]:
            with open(file, "r") as f:
                first_line = f.readline()
            if any(name in first_line for name in self.signal_names):
                data = self._load_csv_with_header(file)
            else:
                data = self._load_txt(file)

        if data.ndim > 1 and len(self.signal_names) == 1:
            data = data.flatten()

        if self.normalize:
            data = self._normalize_signals(data)

        if file:
            self.metadata[index] = self.modality_type.create_ts_metadata(
                self.signal_names, data, self.sampling_rate
            )
        else:
            for i, index in enumerate(self.indices):
                self.metadata[str(index)] = self.modality_type.create_ts_metadata(
                    self.signal_names, data[i], self.sampling_rate
                )
        self.data.append(data)

    def _normalize_signals(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            mean = np.mean(data)
            std = np.std(data)
            return (data - mean) / (std + 1e-8)
        else:
            for i in range(data.shape[1]):
                mean = np.mean(data[:, i])
                std = np.std(data[:, i])
                data[:, i] = (data[:, i] - mean) / (std + 1e-8)
            return data

    def _load_npy(self, file: str) -> np.ndarray:
        data = np.load(file).astype(self._data_type)
        return data

    def _load_txt(self, file: str) -> np.ndarray:
        data = np.loadtxt(file).astype(self._data_type)
        return data

    def _load_txt_with_header(self, file: str) -> np.ndarray:
        with open(file, "r") as f:
            header = f.readline().strip().split()

        col_indices = [
            header.index(name) for name in self.signal_names if name in header
        ]
        data = np.loadtxt(file, dtype=self._data_type, skiprows=1, usecols=col_indices)
        return data

    def _load_csv_with_header(self, file: str, delimiter: str = None) -> np.ndarray:
        import pandas as pd

        if delimiter is None:
            with open(file, "r") as f:
                sample = f.read(1024)
            if "," in sample:
                delimiter = ","
            elif "\t" in sample:
                delimiter = "\t"
            else:
                delimiter = " "
        df = pd.read_csv(file, delimiter=delimiter)

        selected = [name for name in self.signal_names if name in df.columns]
        data = df[selected].to_numpy(dtype=self._data_type)
        return data

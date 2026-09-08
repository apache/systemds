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


def _materialize_timestamps(frequency, sample_length, start_datetime=None):
    start_time = (
        start_datetime
        if start_datetime is not None
        else np.datetime64("1970-01-01T00:00:00.000000")
    )
    time_increment = 1 / frequency
    time_increments_array = np.arange(sample_length) * np.timedelta64(
        int(time_increment * 1e6)
    )
    timestamps = start_time + time_increments_array

    return timestamps.astype(np.int64)


class DerivedTimestamps(np.ndarray):

    def __new__(cls, frequency, sample_length, start_datetime=None):
        obj = _materialize_timestamps(frequency, sample_length, start_datetime).view(
            cls
        )
        obj._generator = (frequency, sample_length, start_datetime)
        return obj

    def __array_finalize__(self, obj):
        self._generator = None

    def __reduce__(self):
        generator = getattr(self, "_generator", None)
        if generator is not None:
            return (DerivedTimestamps, generator)
        return super().__reduce__()

    def __deepcopy__(self, memo):
        generator = getattr(self, "_generator", None)
        if generator is not None:
            return DerivedTimestamps(*generator)
        return super().__deepcopy__(memo)


def create_timestamps(frequency, sample_length, start_datetime=None):
    return DerivedTimestamps(frequency, sample_length, start_datetime)


def calculate_new_frequency(new_length, old_length, old_frequency):
    duration = old_length / old_frequency
    new_frequency = new_length / duration
    return new_frequency


def get_shape(metadata):
    return len(metadata[0]["data_layout"]["shape"])

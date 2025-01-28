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

import pickle

import librosa
import numpy as np
from systemds.scuro.representations.utils import pad_sequences
import matplotlib.pyplot as plt
from systemds.scuro.representations.unimodal import UnimodalRepresentation


class MelSpectrogram(UnimodalRepresentation):
    def __init__(self, avg=True, output_file=None):
        super().__init__("MelSpectrogram")
        self.avg = avg
        self.output_file = output_file

    def transform(self, data):
        result = []
        max_length = 0
        for sample in data:
            S = librosa.feature.melspectrogram(
                y=sample, sr=22050
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            if S_dB.shape[-1] > max_length:
                max_length = S_dB.shape[-1]
            result.append(S_dB.T)

        # r = []
        # for elem in result:
        #     d = pad_sequences(elem, maxlen=max_length, dtype="float32")
        #     r.append(d)

        # np_array_r = np.array(r) if not self.avg else np.mean(np.array(r), axis=1)
        #
        # if self.output_file is not None:
        #     data = []
        #     for i in range(0, np_array_r.shape[0]):
        #         data.append(np_array_r[i])
        #     with open(self.output_file, "wb") as file:
        #         pickle.dump(data, file)

        return result
    
    
    def plot_spectrogram(self, spectrogram):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel', sr=22050, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.savefig('spectrogram.jpg')

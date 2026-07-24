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
from scipy import signal

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.modality.transformed import TransformedModality


class FrequencyRepresentation(UnimodalRepresentation):
    def __init__(
        self, fft_top_k=10, use_spectrogram=False, wavelet=None, output_file=None
    ):
        super().__init__(
            "FrequencyRepresentation", ModalityType.EMBEDDING, self._get_parameters()
        )
        self.fft_top_k = fft_top_k
        self.use_spectrogram = use_spectrogram
        self.wavelet = wavelet
        self.output_file = output_file

    def _get_parameters(self):
        return {
            "fft_top_k": [5, 10, 20],
            "use_spectrogram": [True, False],
            "wavelet": [None, "db4", "haar", "coif1"],
        }

    def transform(self, modality, aggregation=None):
        x = modality.data
        output = []

        fft_vals = np.fft.fft(x)
        fft_freqs = np.fft.fftfreq(len(fft_vals))
        idxs = np.argsort(np.abs(fft_vals))[-self.fft_top_k :]
        fft_feats = np.stack(
            [np.abs(fft_vals[idxs]), fft_freqs[idxs], np.angle(fft_vals[idxs])], axis=1
        )
        output.append(fft_feats.flatten())

        if self.use_spectrogram:
            f, t, Sxx = signal.spectrogram(x)
            output.append(np.mean(Sxx, axis=1))
            output.append(np.std(Sxx, axis=1))

        if self.wavelet is not None:
            import pywt

            coeffs = pywt.wavedec(x, self.wavelet)
            output.append(np.concatenate([c.flatten() for c in coeffs[:3]]))

        transformed_modality = TransformedModality(
            modality, self, ModalityType.EMBEDDING
        )
        transformed_modality.data = np.array(output)
        return transformed_modality

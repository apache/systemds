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
from functools import reduce
from operator import or_

import numpy as np

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.utils import pad_sequences
from systemds.scuro.representations.window_aggregation import WindowAggregation


class JoinedTransformedModality(Modality):

    def __init__(self, left_modality, right_modality, transformation):
        """
        Parent class of the different Modalities (unimodal & multimodal)
        :param transformation: Representation to be applied on the modality
        """
        super().__init__(
            reduce(or_, [left_modality.modality_type], right_modality.modality_type),
            data_type=left_modality.data_type,
        )
        self.transformation = transformation
        self.left_modality = left_modality
        self.right_modality = right_modality

    def combine(self, fusion_method):
        """
        Combines two or more modalities with each other using a dedicated fusion method
        :param other: The modality to be combined
        :param fusion_method: The fusion method to be used to combine modalities
        """
        modalities = [self.left_modality, self.right_modality]
        self.data = []
        for i in range(0, len(self.left_modality.data)):
            self.data.append([])
            for j in range(0, len(self.left_modality.data[i])):
                self.data[i].append([])
                fused = np.concatenate(
                    [self.left_modality.data[i][j], self.right_modality.data[i][j]],
                    axis=0,
                )
                self.data[i][j] = fused
        # self.data = fusion_method.transform(modalities)

        for i, instance in enumerate(
            self.data
        ):  # TODO: only if the layout is list_of_lists_of_numpy_array
            r = []
            [r.extend(l) for l in instance]
            self.data[i] = np.array(r)
        self.data = pad_sequences(self.data)
        return self

    def window_aggregation(self, window_size, aggregation):
        w = WindowAggregation(window_size, aggregation)
        self.left_modality.data = w.execute(self.left_modality)
        self.right_modality.data = w.execute(self.right_modality)
        return self

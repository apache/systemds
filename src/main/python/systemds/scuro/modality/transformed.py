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

from systemds.scuro.modality.joined import JoinedModality
from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.window import WindowAggregation


class TransformedModality(Modality):

    def __init__(self, modality_type, transformation, metadata):
        """
        Parent class of the different Modalities (unimodal & multimodal)
        :param modality_type: Type of the original modality(ies)
        :param transformation: Representation to be applied on the modality
        """
        super().__init__(modality_type, metadata)
        self.transformation = transformation
        self.data = []

    def copy_from_instance(self):
        return type(self)(self.modality_type, self.transformation, self.metadata)

    def join(self, right, join_condition):
        chunked_execution = False
        if type(right).__name__.__contains__("Unimodal"):
            if right.data_loader.chunk_size:
                chunked_execution = True
            elif right.data is None or len(right.data) == 0:
                right.extract_raw_data()

        joined_modality = JoinedModality(
            reduce(or_, [right.modality_type], self.modality_type),
            self,
            right,
            join_condition,
            chunked_execution,
        )

        if not chunked_execution:
            joined_modality.execute(0)

        return joined_modality

    def window(self, windowSize, aggregationFunction, fieldName=None):
        transformed_modality = TransformedModality(
            self.modality_type, "window", self.metadata
        )
        w = WindowAggregation(windowSize, aggregationFunction)
        transformed_modality.data = w.window(self)

        return transformed_modality

    def apply_representation(self, representation, aggregation):
        new_modality = representation.transform(self)

        if aggregation:
            new_modality.data = aggregation.window(new_modality)

        new_modality.update_metadata()
        return new_modality

    def combine(self, other, fusion_method):
        """
        Combines two or more modalities with each other using a dedicated fusion method
        :param other: The modality to be combined
        :param fusion_method: The fusion method to be used to combine modalities
        """
        fused_modality = TransformedModality(
            reduce(or_, (o.modality_type for o in other), self.modality_type),
            fusion_method,
            self.metadata,
        )
        modalities = [self]
        modalities.extend(other)
        fused_modality.data = fusion_method.transform(modalities)

        return fused_modality

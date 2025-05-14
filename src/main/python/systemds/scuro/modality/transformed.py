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

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.joined import JoinedModality
from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.window import WindowAggregation


class TransformedModality(Modality):

    def __init__(self, modality_type, transformation, modality_id, metadata):
        """
        Parent class of the different Modalities (unimodal & multimodal)
        :param modality_type: Type of the original modality(ies)
        :param transformation: Representation to be applied on the modality
        """
        super().__init__(modality_type, modality_id, metadata)
        self.transformation = transformation

    def copy_from_instance(self):
        return type(self)(
            self.modality_type, self.transformation, self.modality_id, self.metadata
        )

    def join(self, right, join_condition):
        chunked_execution = False
        if type(right).__name__.__contains__("Unimodal"):
            if right.data_loader.chunk_size:
                chunked_execution = True
            elif not right.has_data():
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
            joined_modality.joined_right.update_metadata()

        return joined_modality

    def window(self, windowSize, aggregation):
        transformed_modality = TransformedModality(
            self.modality_type, "window", self.modality_id, self.metadata
        )
        w = WindowAggregation(windowSize, aggregation)
        transformed_modality.data = w.execute(self)

        return transformed_modality

    def context(self, context_operator):
        transformed_modality = TransformedModality(
            self.modality_type, context_operator.name, self.modality_id, self.metadata
        )

        transformed_modality.data = context_operator.execute(self)
        return transformed_modality

    def apply_representation(self, representation):
        new_modality = representation.transform(self)
        new_modality.update_metadata()
        return new_modality

    def combine(self, other, fusion_method):
        """
        Combines two or more modalities with each other using a dedicated fusion method
        :param other: The modality to be combined
        :param fusion_method: The fusion method to be used to combine modalities
        """
        fused_modality = TransformedModality(
            ModalityType.EMBEDDING,
            fusion_method,
            self.modality_id,
            self.metadata,
        )
        modalities = [self]
        modalities.extend(other)
        fused_modality.data = fusion_method.transform(modalities)

        return fused_modality

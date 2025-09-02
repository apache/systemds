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
from typing import Union, List

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.joined import JoinedModality
from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.window_aggregation import WindowAggregation
import time
import copy


class TransformedModality(Modality):

    def __init__(
        self, modality, transformation, new_modality_type=None, self_contained=True
    ):
        """
        Parent class of the different Modalities (unimodal & multimodal)
        :param modality_type: Type of the original modality(ies)
        :param transformation: Representation to be applied on the modality
        """
        if new_modality_type is None:
            new_modality_type = modality.modality_type

        metadata = modality.metadata.copy() if modality.metadata is not None else None
        super().__init__(
            new_modality_type, modality.modality_id, metadata, modality.data_type
        )
        self.transformation = None
        self.self_contained = (
            self_contained and transformation.self_contained
            if isinstance(transformation, TransformedModality)
            else True
        )
        self.add_transformation(transformation, modality)

        if modality.__class__.__name__ == "UnimodalModality":
            for k, v in self.metadata.items():
                if "attention_masks" in v:
                    del self.metadata[k]["attention_masks"]

    def add_transformation(self, transformation, modality):
        if (
            transformation.__class__.__bases__[0].__name__ == "Fusion"
            and modality.transformation[0].__class__.__bases__[0].__name__ != "Fusion"
        ):
            self.transformation = []
        else:
            self.transformation = (
                []
                if type(modality).__name__ != "TransformedModality"
                else copy.deepcopy(modality.transformation)
            )
        self.transformation.append(transformation)

    def copy_from_instance(self):
        return type(self)(self, self.transformation)

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

    def window_aggregation(self, windowSize, aggregation):
        w = WindowAggregation(windowSize, aggregation)
        transformed_modality = TransformedModality(
            self, w, self_contained=self.self_contained
        )
        start = time.time()
        transformed_modality.data = w.execute(self)
        transformed_modality.transform_time = time.time() - start
        return transformed_modality

    def context(self, context_operator):
        transformed_modality = TransformedModality(
            self, context_operator, self_contained=self.self_contained
        )
        start = time.time()
        transformed_modality.data = context_operator.execute(self)
        transformed_modality.transform_time = time.time() - start
        return transformed_modality

    def apply_representation(self, representation):
        start = time.time()
        new_modality = representation.transform(self)
        new_modality.update_metadata()
        new_modality.transform_time = time.time() - start
        new_modality.self_contained = representation.self_contained
        return new_modality

    def combine(self, other: Union[Modality, List[Modality]], fusion_method):
        """
        Combines two or more modalities with each other using a dedicated fusion method
        :param other: The modality to be combined
        :param fusion_method: The fusion method to be used to combine modalities
        """
        fused_modality = TransformedModality(
            self, fusion_method, ModalityType.EMBEDDING
        )
        modalities = [self]
        if isinstance(other, list):
            modalities.extend(other)
        else:
            modalities.append(other)
        fused_modality.data = fusion_method.transform(modalities)

        return fused_modality

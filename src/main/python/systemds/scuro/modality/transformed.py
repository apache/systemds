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

from systemds.scuro.modality.modality import Modality


class TransformedModality(Modality):

    def __init__(self, modality_type, transformation):
        """
        Parent class of the different Modalities (unimodal & multimodal)
        :param modality_type: Type of the original modality(ies)
        :param transformation: Representation to be applied on the modality
        """
        super().__init__(modality_type)
        self.transformation = transformation
        self.data = []

    def combine(self, other, fusion_method):
        """
        Combines two or more modalities with each other using a dedicated fusion method
        :param other: The modality to be combined
        :param fusion_method: The fusion method to be used to combine modalities
        """
        fused_modality = TransformedModality(
            reduce(or_, (o.type for o in other), self.type), fusion_method
        )
        modalities = [self]
        modalities.extend(other)
        fused_modality.data = fusion_method.transform(modalities)

        return fused_modality

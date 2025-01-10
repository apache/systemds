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
from aligner.alignment_strategy import AlignmentStrategy
from modality.modality import Modality
from modality.representation import Representation
from aligner.similarity_measures import Measure


class Alignment:
    def __init__(
        self,
        modality_a: Modality,
        modality_b: Modality,
        strategy: AlignmentStrategy,
        similarity_measure: Measure,
    ):
        """
        Defines the core of the library where the alignment of two modalities is performed
        :param modality_a: first modality
        :param modality_b: second modality
        :param strategy: the alignment strategy used in the alignment process
        :param similarity_measure: the similarity measure used to check the score of the alignment
        """
        self.modality_a = modality_a
        self.modality_b = modality_b
        self.strategy = strategy
        self.similarity_measure = similarity_measure

    def align_modalities(self) -> Modality:
        return Modality(Representation())

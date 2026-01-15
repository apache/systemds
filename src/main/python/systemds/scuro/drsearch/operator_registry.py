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
from typing import Union, List

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.representation import Representation


class Registry:
    """
    A registry for all representations per modality.
    The representations are stored in a dictionary where a specific modality type is the key.
    Implemented as a singleton.
    """

    _instance = None
    _representations = {}
    _context_operators = {}
    _fusion_operators = []
    _text_context_operators = []
    _video_context_operators = []
    _dimensionality_reduction_operators = {}

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            for m_type in ModalityType:
                cls._representations[m_type] = []
        return cls._instance

    def set_fusion_operators(self, fusion_operators):
        if isinstance(fusion_operators, list):
            self._context_operators = fusion_operators
        else:
            self._fusion_operators = [fusion_operators]

    def set_representations(self, modality_type, representations):
        if isinstance(representations, list):
            self._representations[modality_type] = representations
        else:
            self._representations[modality_type] = [representations]

    def add_representation(
        self, representation: Representation, modality: ModalityType
    ):
        self._representations[modality].append(representation)

    def add_context_operator(self, context_operator, modality_type):
        if not isinstance(modality_type, list):
            modality_type = [modality_type]
        for m_type in modality_type:
            if not m_type in self._context_operators.keys():
                self._context_operators[m_type] = []
            self._context_operators[m_type].append(context_operator)

    def add_fusion_operator(self, fusion_operator):
        self._fusion_operators.append(fusion_operator)

    def add_dimensionality_reduction_operator(
        self, dimensionality_reduction_operator, modality_type
    ):
        if not isinstance(modality_type, list):
            modality_type = [modality_type]
        for m_type in modality_type:
            if not m_type in self._dimensionality_reduction_operators.keys():
                self._dimensionality_reduction_operators[m_type] = []
            self._dimensionality_reduction_operators[m_type].append(
                dimensionality_reduction_operator
            )

    def get_representations(self, modality: ModalityType):
        return self._representations[modality]

    def get_not_self_contained_representations(self, modality: ModalityType):
        reps = []
        for rep in self.get_representations(modality):
            if not rep().self_contained:
                reps.append(rep)
        return reps

    def get_context_operators(self, modality_type):
        return self._context_operators[modality_type]

    def get_dimensionality_reduction_operators(self, modality_type):
        return self._dimensionality_reduction_operators[modality_type]

    def get_fusion_operators(self):
        return self._fusion_operators

    def get_fusion_operator_by_name(self, fusion_name):
        for fusion in self._fusion_operators:
            if fusion.__name__ == fusion_name:
                return fusion
        return None

    def get_representation_by_name(self, representation_name, modality_type):
        for representation in self._context_operators:
            if representation.__name__ == representation_name:
                return representation, True

        if modality_type is not None:
            for representation in self._representations[modality_type]:
                if representation.__name__ == representation_name:
                    return representation, False

        return None, False


def register_representation(modalities: Union[ModalityType, List[ModalityType]]):
    """
    Decorator to register representation for a specific modality.
    :param modalities: The modalities for which the representation is to be registered
    """
    if isinstance(modalities, ModalityType):
        modalities = [modalities]

    def decorator(cls):
        for modality in modalities:
            if modality not in ModalityType:
                raise f"Modality {modality} not in ModalityTypes please add it to constants.py ModalityTypes first!"

            Registry().add_representation(cls, modality)
        return cls

    return decorator


def register_dimensionality_reduction_operator(modality_type):
    """
    Decorator to register a dimensionality reduction operator.
    """

    def decorator(cls):
        Registry().add_dimensionality_reduction_operator(cls, modality_type)
        return cls

    return decorator


def register_context_operator(modality_type):
    """
    Decorator to register a context operator.

    @param modality_type: The modality type for which the context operator is to be registered
    """

    def decorator(cls):
        Registry().add_context_operator(cls, modality_type)
        return cls

    return decorator


def register_fusion_operator():
    """
    Decorator to register a fusion operator.
    """

    def decorator(cls):
        Registry().add_fusion_operator(cls)
        return cls

    return decorator

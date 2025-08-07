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
import itertools
import random
from typing import List

from systemds.scuro.drsearch.task import Task
from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.representation import Representation

import warnings

warnings.filterwarnings("ignore")


def get_modalities_by_name(modalities, name):
    for modality in modalities:
        if modality.name == name:
            return modality

    raise "Modality " + name + "not in modalities"


class DRSearch:
    def __init__(
        self,
        modalities: List[Modality],
        task: Task,
        representations: List[Representation],
    ):
        """
        The DRSearch primitive finds the best uni- or multimodal data representation for the given modalities for
        a specific task
        :param modalities: List of uni-modal modalities
        :param task: custom task
        :param representations: List of representations to be evaluated
        """
        self.modalities = modalities
        self.task = task
        self.representations = representations
        self.scores = {}
        self.best_modalities = None
        self.best_representation = None
        self.best_score = -1

    def set_best_params(
        self,
        representation: Representation,
        scores: List[float],
        modality_names: List[str],
    ):
        """
        Updates the best parameters for given modalities, representation, and score
        :param representation: The representation used to retrieve the current score
        :param scores: achieved train/test scores for the set of modalities and representation
        :param modality_names: List of modality names used in this setting
        :return:
        """

        # check if modality name is already in dictionary
        if "_".join(modality_names) not in list(self.scores.keys()):
            # if not add it to dictionary
            self.scores["_".join(modality_names)] = {}

        # set score for representation
        self.scores["_".join(modality_names)][representation] = scores

        # compare current score with best score
        if scores[1] > self.best_score:
            self.best_score = scores[1]
            self.best_representation = representation
            self.best_modalities = modality_names

    def reset_best_params(self):
        self.best_score = -1
        self.best_modalities = None
        self.best_representation = None
        self.scores = {}

    def fit_random(self, seed=-1):
        """
        This method randomly selects a modality or combination of modalities and representation
        """
        if seed != -1:
            random.seed(seed)

        modalities = []
        for M in range(1, len(self.modalities) + 1):
            for combination in itertools.combinations(self.modalities, M):
                modalities.append(combination)

        modality_combination = random.choice(modalities)
        representation = random.choice(self.representations)

        modality = modality_combination[0].combine(
            list(modality_combination[1:]), representation
        )

        scores = self.task.run(modality.data)
        self.set_best_params(representation, scores, modality.get_modality_names())

        return self.best_representation, self.best_score, self.best_modalities

    def fit_enumerate_all(self):
        """
        This method finds the best representation out of a given List of uni-modal modalities and
        representations
        :return: The best parameters found in the search procedure
        """

        for M in range(1, len(self.modalities) + 1):
            for combination in itertools.combinations(self.modalities, M):
                for representation in self.representations:
                    modality = combination[0]
                    if len(combination) > 1:
                        modality = combination[0].combine(
                            list(combination[1:]), representation
                        )

                    scores = self.task.run(modality.data)
                    self.set_best_params(
                        representation,
                        scores,
                        modality.get_modality_names(),
                    )

        return self.best_representation, self.best_score, self.best_modalities

    def transform(self, modalities: List[Modality]):
        """
        The transform method takes a list of uni-modal modalities and creates an aligned representation
        by using the best parameters found during the fitting step
        :param modalities: List of uni-modal modalities
        :return: aligned data
        """

        if self.best_score == -1:
            raise "Please fit representations first!"

        used_modalities = []

        for modality_name in self.best_modalities:
            used_modalities.append(get_modalities_by_name(modalities, modality_name))

        modality = used_modalities[0].combine(
            used_modalities[1:], self.best_representation
        )

        return modality.data

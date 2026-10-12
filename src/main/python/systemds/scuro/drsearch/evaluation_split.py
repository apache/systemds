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
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationSplit:
    """Global instance IDs used to fit and evaluate one pipeline."""

    fit_indices: tuple
    evaluation_indices: tuple
    name: str = ""

    def __post_init__(self):
        fit_indices = tuple(int(i) for i in self.fit_indices)
        evaluation_indices = tuple(int(i) for i in self.evaluation_indices)

        if len(set(fit_indices)) != len(fit_indices):
            raise ValueError("fit_indices contains duplicate instance IDs")
        if len(set(evaluation_indices)) != len(evaluation_indices):
            raise ValueError("evaluation_indices contains duplicate instance IDs")
        overlap = set(fit_indices).intersection(evaluation_indices)
        if overlap:
            raise ValueError(
                "Fit and evaluation instance IDs must be disjoint: "
                f"{sorted(overlap)}"
            )

        object.__setattr__(self, "fit_indices", fit_indices)
        object.__setattr__(self, "evaluation_indices", evaluation_indices)

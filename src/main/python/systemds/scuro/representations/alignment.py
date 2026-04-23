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
from abc import ABC, abstractmethod
from collections import defaultdict
import copy


@dataclass
class Match:
    primary: int
    secondary: int
    distance: float


class Alignment(ABC):
    def __init__(self, name):
        self.name = name

    def execute(self, primary_modality, secondary_modality):
        primary_descriptor_collections = self._batch_compute_descriptors(
            primary_modality
        )
        secondary_descriptor_collections = self._batch_compute_descriptors(
            secondary_modality
        )

        matches = []

        for p, p_collection in enumerate(primary_descriptor_collections):
            stats = defaultdict(
                lambda: {
                    "count": 0,
                    "total_distance": 0.0,
                    "best_distance": float("inf"),
                }
            )

            for p_desc in p_collection:
                best_secondary = None
                best_dist = float("inf")

                for s, s_collection in enumerate(secondary_descriptor_collections):
                    for s_desc in s_collection:
                        dist = self.compare(p_desc, s_desc)

                        if dist < best_dist:
                            best_dist = dist
                            best_secondary = s

                if best_secondary is not None:
                    stats[best_secondary]["count"] += 1
                    stats[best_secondary]["total_distance"] += best_dist
                    stats[best_secondary]["best_distance"] = min(
                        stats[best_secondary]["best_distance"], best_dist
                    )

            if not stats:
                matches.append(Match(p, None, float("inf")))
                continue

            best_match = min(
                stats.items(),
                key=lambda item: (
                    -item[1]["count"],  # mehr Votes ist besser
                    item[1]["total_distance"],  # kleinere Gesamtdistanz ist besser
                    item[1]["best_distance"],  # optional weiterer Tie-Breaker
                ),
            )[0]

            result_distance = (
                stats[best_match]["total_distance"] / stats[best_match]["count"]
            )

            matches.append(Match(p, best_match, result_distance))

        return matches

    @staticmethod
    def apply_matching(alignment, secondary_modality):
        aligned_modality = copy.deepcopy(secondary_modality)
        aligned_modality.data = [None] * len(alignment)
        aligned_modality.metadata = [None] * len(alignment)

        for match in alignment:
            aligned_modality.data[match.primary] = secondary_modality.data[
                match.secondary
            ]
            aligned_modality.metadata[match.primary] = secondary_modality.metadata[
                match.secondary
            ]

        return aligned_modality

    def _batch_compute_descriptors(self, modality):
        descriptors = []

        if modality.data_loader.chunk_size:
            modality.data_loader.reset()
            while modality.data_loader.next_chunk < modality.data_loader.num_chunks:
                modality.extract_raw_data()
                for d in modality.data:
                    descriptors.append(self.compute_descriptor(d))
        else:
            if not modality.has_data():
                modality.extract_raw_data()
                for d in modality.data:
                    descriptors.append(self.compute_descriptor(d))

        return descriptors

    @abstractmethod
    def compute_descriptor(self, segment):
        pass

    @abstractmethod
    def compare(self, a, b):
        pass

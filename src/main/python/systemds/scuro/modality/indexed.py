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

from systemds.scuro.modality.modality import Modality


@dataclass(frozen=True)
class IndexedModality:
    """A modality partition paired with its global instance IDs."""

    modality: Modality
    instance_ids: tuple

    def __post_init__(self):
        instance_ids = tuple(int(i) for i in self.instance_ids)
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("instance_ids contains duplicate instance IDs")

        if self.modality.has_data():
            num_instances = len(self.modality.data)
        elif self.modality.has_metadata():
            num_instances = len(self.modality.metadata)
        else:
            raise ValueError(
                "IndexedModality requires materialized data or instance metadata"
            )

        if len(instance_ids) != num_instances:
            raise ValueError(
                f"Expected {num_instances} instance IDs, got {len(instance_ids)}"
            )

        object.__setattr__(self, "instance_ids", instance_ids)

    def __len__(self):
        return len(self.instance_ids)

    def subset(self, instance_ids) -> "IndexedModality":
        requested_ids = tuple(int(i) for i in instance_ids)
        positions = {instance_id: i for i, instance_id in enumerate(self.instance_ids)}
        unknown_ids = [i for i in requested_ids if i not in positions]
        if unknown_ids:
            raise KeyError(f"Unknown instance IDs: {unknown_ids}")

        local_indices = [positions[i] for i in requested_ids]
        return IndexedModality(
            self.modality.subset(local_indices),
            requested_ids,
        )

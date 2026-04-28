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
import json
import os
import numpy as np
from typing import Tuple
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.dataloader.base_loader import BaseLoader
from typing import Optional, List, Union


@dataclass
class JSONStats:
    num_instances: int
    max_length: int
    avg_length: float
    max_words: int
    avg_words: float
    output_shape: Tuple[int]


class JSONLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        field: str,  # TODO: make this a list so it is easier to get multiple fields from a json file. (i.e. Mustard: context + sentence)
        data_type: Union[np.dtype, str] = str,
        chunk_size: Optional[int] = None,
        ext: str = ".json",
    ):
        super().__init__(
            source_path, indices, data_type, chunk_size, ModalityType.TEXT, ext
        )
        self.field = field
        self.stats = self.get_stats(source_path)

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        self.file_sanity_check(file)
        with open(file) as f:
            json_file = json.load(f)

            if isinstance(index, str):
                index = [index]
            for idx in index:
                try:
                    text = json_file[idx][self.field]
                except:
                    text = json_file[self.field]

                text = " ".join(text) if isinstance(text, list) else text
                self.data.append(text)
                self.metadata.append(
                    self.modality_type.create_metadata(len(text), text) | json_file[idx]
                )

    def get_stats(self, source_path: str):
        self.file_sanity_check(source_path)
        num_instances = 0
        max_length = 0
        avg_length = 0
        max_words = 0
        avg_words = 0
        if os.path.isfile(source_path):
            with open(source_path) as f:
                json_file = json.load(f)
                for id in self.indices:
                    try:
                        text = json_file[id][self.field]
                    except:
                        text = json_file[self.field]

                    text = " ".join(text) if isinstance(text, list) else text
                    num_instances += 1
                    max_length = max(max_length, len(text))
                    avg_length += len(text)
        else:
            for id in self.indices:
                with open(os.path.join(source_path, f"{id}{self._ext}")) as f:
                    json_file = json.load(f)
                    text = json_file[self.field]

                    text = " ".join(text) if isinstance(text, list) else text
                    num_instances += 1
                    max_length = max(max_length, len(text))  # number of characters
                    max_words = max(max_words, len(text.split()))
                    avg_words += len(text.split())
                    avg_length += len(text)

            avg_length /= num_instances
            avg_words /= num_instances
        return JSONStats(
            num_instances,
            max_length,
            avg_length,
            max_words,
            avg_words,
            (max_length,),
        )

    def estimate_peak_memory_bytes(self) -> dict:
        s = self.stats
        n = max(1, s.num_instances)

        avg_len = s.avg_length / n if s.avg_length > s.max_length else s.avg_length
        avg_len = max(1.0, avg_len)

        bytes_per_char = 2
        str_overhead = 49
        ptr_size = 8
        list_header = 56
        list_overalloc = 1.125
        metadata_per_instance = 192

        resident_strings = n * (str_overhead + bytes_per_char * avg_len)
        resident_list = list_header + int(ptr_size * n * list_overalloc)
        resident_metadata = n * metadata_per_instance
        resident_total = resident_strings + resident_list + resident_metadata

        parse_factor = 2.0
        transient_parse = parse_factor * (n * bytes_per_char * avg_len)

        cpu_peak = int(resident_total + transient_parse)
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}

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
import os
from systemds.scuro.dataloader.base_loader import BaseLoader
from typing import Optional, Pattern, List, Union
from systemds.scuro.modality.type import ModalityType
import re


@dataclass
class TextStats:
    num_instances: int
    max_length: int
    avg_length: float
    max_words: int
    avg_words: float
    output_shape: tuple


class TextLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        data_type: str = str,
        chunk_size: Optional[int] = None,
        prefix: Optional[Pattern[str]] = None,
    ):
        super().__init__(source_path, indices, data_type, chunk_size, ModalityType.TEXT)
        self.prefix = prefix
        self.stats = self.get_stats(source_path)

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        self.file_sanity_check(file)
        with open(file) as text_file:
            for i, line in enumerate(text_file):
                if self.prefix:
                    line = re.sub(self.prefix, "", line)
                line = line.replace("\n", "")
                self.metadata.append(
                    self.modality_type.create_metadata(len(line.split()), line)
                )
                self.data.append(line)

    def get_stats(self, source_path: str):
        num_instances = 0
        max_length = 0
        avg_length = 0
        max_words = 0
        avg_words = 0
        for file in os.listdir(source_path):
            self.file_sanity_check(source_path + file)
            with open(source_path + file) as text_file:
                for line in text_file:
                    if self.prefix:
                        line = re.sub(self.prefix, "", line)
                    line = line.replace("\n", "")
                    length = len(line.split())
                    num_instances += 1
                    max_length = max(max_length, length)
                    avg_length += length
                    max_words = max(max_words, len(line.split()))
                    avg_words += len(line.split())
            avg_length /= num_instances
            avg_words /= num_instances
        return TextStats(
            num_instances, max_length, avg_length, max_words, avg_words, (max_length,)
        )

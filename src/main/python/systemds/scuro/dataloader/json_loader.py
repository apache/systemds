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
import json

from systemds.scuro.dataloader.base_loader import BaseLoader
from typing import Optional, List


class JSONLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        field: str,
        chunk_size: Optional[int] = None,
    ):
        super().__init__(source_path, indices, chunk_size)
        self.field = field

    def extract(self, file: str, indices: List[str]):
        self.file_sanity_check(file)
        with open(file) as f:
            json_file = json.load(f)
            for idx in indices:
                self.data.append(json_file[idx][self.field])

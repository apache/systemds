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
from enum import Flag, auto


class ModalitySchemas:
    TEXT_SCHEMA = {"type": "string", "length": "int"}

    AUDIO_SCHEMA = {
        "timestamp": "array",
        "type": "float32",
        "sample_rate": "integer",
        "length": "integer",
    }

    VIDEO_SCHEMA = {
        "timestamp": "array",
        "type": "object",
        "fps": "integer",
        "length": "integer",
        "width": "integer",
        "height": "integer",
        "num_channels": "integer",
    }

    @classmethod
    def get(cls, name):
        return getattr(cls, f"{name}_SCHEMA", None)

    @classmethod
    def add_schema(cls, name, schema):
        setattr(cls, f"{name}_SCHEMA", schema)


class ModalityType(Flag):
    TEXT = auto()
    AUDIO = auto()
    VIDEO = auto()

    def get_schema(self):
        return ModalitySchemas.get(self.name)

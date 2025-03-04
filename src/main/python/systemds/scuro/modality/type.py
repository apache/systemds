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
from systemds.scuro.utils.schema_helpers import (
    calculate_new_frequency,
    create_timestamps,
)


# TODO: needs a way to define if data comes from a dataset with multiple instances or is like a streaming scenario where we only have one instance
# right now it is a list of instances (if only one instance the list would contain only a single item)
class ModalitySchemas:
    TEXT_SCHEMA = {"type": "string", "length": "int"}

    AUDIO_SCHEMA = {
        "timestamp": "array",
        "data_layout": {"type": "?", "representation": "?"},
        "sample_rate": "integer",
        "length": "integer",
    }

    VIDEO_SCHEMA = {
        "timestamp": "array",
        "data_layout": {"type": "?", "representation": "?"},
        "fps": "integer",
        "length": "integer",
        "width": "integer",
        "height": "integer",
        "num_channels": "integer",
    }

    _metadata_handlers = {}

    @classmethod
    def get(cls, name):
        return getattr(cls, f"{name}_SCHEMA", None)

    @classmethod
    def add_schema(cls, name, schema):
        setattr(cls, f"{name}_SCHEMA", schema)

    @classmethod
    def register_metadata_handler(cls, name):
        def decorator(metadata_handler):
            cls._metadata_handlers[name] = metadata_handler
            return metadata_handler

        return decorator

    @classmethod
    def update_metadata(cls, name, md, data):
        mdHandler = cls._metadata_handlers.get(name)
        if mdHandler:
            return mdHandler(md, data)

    def extract_data(self, data, index):
        if self.get("data_layout").get("representation") == "list_array":
            return data[index]
        else:
            return data[index]


@ModalitySchemas.register_metadata_handler("AUDIO")
def handle_audio_metadata(md, data):
    new_frequency = calculate_new_frequency(len(data), md["length"], md["sample_rate"])
    md.update(
        {
            "length": len(data),
            "sample_rate": new_frequency,
            "timestamp": create_timestamps(new_frequency, len(data)),
        }
    )
    return md


@ModalitySchemas.register_metadata_handler("VIDEO")
def handle_video_metadata(md, data):
    new_frequency = calculate_new_frequency(len(data), md["length"], md["fps"])
    md.update(
        {
            "length": len(data),
            "fps": new_frequency,
            "timestamp": create_timestamps(new_frequency, len(data)),
        }
    )
    return md


class ModalityType(Flag):
    TEXT = auto()
    AUDIO = auto()
    VIDEO = auto()

    def get_schema(self):
        return ModalitySchemas.get(self.name)

    def update_metadata(self, md, data):
        return ModalitySchemas.update_metadata(self.name, md, data)

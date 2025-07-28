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
from enum import Flag, auto, Enum
from copy import deepcopy
import numpy as np

from systemds.scuro.utils.schema_helpers import (
    calculate_new_frequency,
    create_timestamps,
)


# TODO: needs a way to define if data comes from a dataset with multiple instances or is like a streaming scenario where we only have one instance
# right now it is a list of instances (if only one instance the list would contain only a single item)
class ModalitySchemas:
    BASE_SCHEMA = {"data_layout": {"type": "?", "representation": "?", "shape": "?"}}

    TEMPORAL_BASE_SCHEMA = {
        **BASE_SCHEMA,
        "timestamp": "array",
        "length": "integer",
        "frequency": "float",
    }

    TEXT_SCHEMA = {**BASE_SCHEMA, "length": "int"}

    EMBEDDING_SCHEMA = {**BASE_SCHEMA, "length": "int"}

    AUDIO_SCHEMA = {**TEMPORAL_BASE_SCHEMA, "frequency": "integer"}

    VIDEO_SCHEMA = {
        **TEMPORAL_BASE_SCHEMA,
        "width": "integer",
        "height": "integer",
        "num_channels": "integer",
    }

    IMAGE_SCHEMA = {
        **BASE_SCHEMA,
        "width": "integer",
        "height": "integer",
        "num_channels": "integer",
    }

    TIMESERIES_SCHEMA = {
        **TEMPORAL_BASE_SCHEMA,
        "num_columns": "integer",
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
        md = cls.update_base_metadata(md, data)
        mdHandler = cls._metadata_handlers.get(name)
        if mdHandler:
            return mdHandler(md, data)

        return md

    @classmethod
    def update_base_metadata(cls, md, data, data_is_single_instance=True):
        data_layout = DataLayout.get_data_layout(data, data_is_single_instance)

        dtype = np.nan
        shape = None
        if data_layout is DataLayout.SINGLE_LEVEL:
            dtype = data.dtype
            shape = data.shape
        elif data_layout is DataLayout.NESTED_LEVEL:
            shape = data[0].shape
            dtype = data[0].dtype

        md["data_layout"].update(
            {"representation": data_layout, "type": dtype, "shape": shape}
        )
        return md

    def extract_data(self, data, index):
        if self.get("data_layout").get("representation") == "list_array":
            return data[index]
        else:
            return data[index]


@ModalitySchemas.register_metadata_handler("AUDIO")
def handle_audio_metadata(md, data):
    new_frequency = calculate_new_frequency(len(data), md["length"], md["frequency"])
    md.update(
        {
            "length": len(data),
            "frequency": new_frequency,
            "timestamp": create_timestamps(new_frequency, len(data)),
        }
    )
    return md


@ModalitySchemas.register_metadata_handler("VIDEO")
def handle_video_metadata(md, data):
    new_frequency = calculate_new_frequency(len(data), md["length"], md["frequency"])
    md.update(
        {
            "length": len(data),
            "frequency": new_frequency,
            "timestamp": create_timestamps(new_frequency, len(data)),
        }
    )
    return md


@ModalitySchemas.register_metadata_handler("IMAGE")
def handle_image_metadata(md, data):
    md.update(
        {
            "width": data.shape[1] if isinstance(data, np.ndarray) else 1,
            "height": data.shape[0] if isinstance(data, np.ndarray) else len(data),
            "num_channels": 1,  # if data.ndim <= 2 else data.shape[2],
        }
    )
    return md


@ModalitySchemas.register_metadata_handler("TIMESERIES")
def handle_timeseries_metadata(md, data):
    new_frequency = calculate_new_frequency(len(data), md["length"], md["frequency"])
    md.update(
        {
            "length": len(data),
            "num_columns": (
                data.shape[1] if isinstance(data, np.ndarray) and data.ndim > 1 else 1
            ),
            "frequency": new_frequency,
            "timestamp": create_timestamps(new_frequency, len(data)),
        }
    )
    return md


@ModalitySchemas.register_metadata_handler("TEXT")
def handle_text_metadata(md, data):
    md.update({"length": len(data)})
    return md


class ModalityType(Flag):
    TEXT = auto()
    AUDIO = auto()
    VIDEO = auto()
    IMAGE = auto()
    TIMESERIES = auto()
    EMBEDDING = auto()

    def get_schema(self):
        return ModalitySchemas.get(self.name)

    def update_metadata(self, md, data):
        return ModalitySchemas.update_metadata(self.name, md, data)
    
    def add_alignment(self, md, alignment_timestamps):
        md["alignment_timestamps"] = alignment_timestamps
        return md
    
    def add_field(self, md, field, data):
        md[field] = data
        return md

    def create_audio_metadata(self, sampling_rate, data):
        md = deepcopy(self.get_schema())
        md = ModalitySchemas.update_base_metadata(md, data, True)
        md["frequency"] = sampling_rate
        md["length"] = data.shape[0]
        md["timestamp"] = create_timestamps(sampling_rate, md["length"])
        return md

    def create_text_metadata(self, length, data):
        md = deepcopy(self.get_schema())
        md["data_layout"]["representation"] = DataLayout.SINGLE_LEVEL
        md["data_layout"]["shape"] = (1, length)
        md["data_layout"]["type"] = str
        md["length"] = length
        return md

    def create_video_metadata(self, frequency, length, width, height, num_channels):
        md = deepcopy(self.get_schema())
        md["frequency"] = frequency
        md["length"] = length
        md["width"] = width
        md["height"] = height
        md["num_channels"] = num_channels
        md["timestamp"] = create_timestamps(frequency, length)
        md["data_layout"]["representation"] = DataLayout.NESTED_LEVEL
        md["data_layout"]["type"] = float
        md["data_layout"]["shape"] = (width, height, num_channels)

        return md


class DataLayout(Enum):
    SINGLE_LEVEL = 1
    NESTED_LEVEL = 2

    @classmethod
    def get_data_layout(cls, data, data_is_single_instance):
        if data is None or len(data) == 0:
            return None

        if data_is_single_instance:
            if isinstance(data, list):
                return DataLayout.NESTED_LEVEL
            elif isinstance(data, np.ndarray):
                return DataLayout.SINGLE_LEVEL

        if isinstance(data[0], list):
            return DataLayout.NESTED_LEVEL
        elif isinstance(data[0], np.ndarray):
            return DataLayout.SINGLE_LEVEL

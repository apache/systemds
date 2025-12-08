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
import torch


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
            if isinstance(data, list):
                dtype = data[0].dtype
                shape = data[0].shape
            elif isinstance(data, np.ndarray):
                dtype = data.dtype
                shape = data.shape
        elif data_layout is DataLayout.NESTED_LEVEL:
            if data_is_single_instance:
                dtype = data.dtype
                shape = data.shape
            else:
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
    PHYSIOLOGICAL = auto()

    _metadata_factory_methods = {
        "TEXT": "create_text_metadata",
        "AUDIO": "create_audio_metadata",
        "VIDEO": "create_video_metadata",
        "IMAGE": "create_image_metadata",
        "TIMESERIES": "create_ts_metadata",
    }

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

    def add_field_for_instances(self, md, field, data):
        for key, value in zip(md.keys(), data):
            md[key].update({field: value})

        return md

    def create_metadata(self, *args, **kwargs):
        if self.name is None or "|" in self.name:
            raise ValueError(
                f"Composite modality types are not supported for metadata creation: {self}"
            )

        factory_methods = type(self)._metadata_factory_methods
        method_name = factory_methods.value.get(self.name)
        if method_name is None:
            raise NotImplementedError(
                f"Metadata creation not implemented for modality type: {self.name}"
            )

        method = getattr(type(self), method_name, None)
        if method is None:
            raise NotImplementedError(
                f"Metadata creation method '{method_name}' not found for {self.name}"
            )

        return method(self, *args, **kwargs)

    def create_audio_metadata(self, sampling_rate, data, is_single_instance=True):
        md = deepcopy(self.get_schema())
        md = ModalitySchemas.update_base_metadata(md, data, is_single_instance)
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

    def create_ts_metadata(
        self, signal_names, data, sampling_rate=None, is_single_instance=True
    ):
        md = deepcopy(self.get_schema())
        md = ModalitySchemas.update_base_metadata(md, data, is_single_instance)
        md["frequency"] = sampling_rate if sampling_rate is not None else 1
        md["length"] = data.shape[0]
        md["signal_names"] = signal_names
        md["timestamp"] = create_timestamps(md["frequency"], md["length"])
        md["is_multivariate"] = len(signal_names) > 1
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
        md["data_layout"]["type"] = np.float32
        md["data_layout"]["shape"] = (width, height, num_channels)
        return md

    def create_image_metadata(self, width, height, num_channels):
        md = deepcopy(self.get_schema())
        md["width"] = width
        md["height"] = height
        md["num_channels"] = num_channels
        md["data_layout"]["representation"] = DataLayout.SINGLE_LEVEL
        md["data_layout"]["type"] = np.float32
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
            if (
                isinstance(data, list)
                or isinstance(data, np.ndarray)
                and data.ndim == 1
            ):
                return DataLayout.SINGLE_LEVEL
            elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
                return DataLayout.NESTED_LEVEL

        if isinstance(data[0], list):
            return DataLayout.NESTED_LEVEL
        elif isinstance(data[0], np.ndarray):
            return DataLayout.SINGLE_LEVEL

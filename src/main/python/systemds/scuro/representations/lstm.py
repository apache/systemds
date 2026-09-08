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
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Any
from systemds.scuro.utils.static_variables import get_device
import numpy as np

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.representations.representation import (
    FitRequirement,
    RepresentationStats,
    stats_bytes,
)

from systemds.scuro.drsearch.operator_registry import register_fusion_operator


@register_fusion_operator()
class LSTM(Fusion):
    fit_requirement = FitRequirement.SUPERVISED

    def __init__(
        self,
        width=128,
        depth=1,
        dropout_rate=0.1,
        learning_rate=0.001,
        epochs=20,
        batch_size=32,
        max_input_dim=16384,
        max_parameter_bytes=1024**3,
        params=None,
    ):
        parameters = {
            "width": [128, 256, 512],
            "depth": [1, 2, 3],
            "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
            "learning_rate": [0.001, 0.0001, 0.01, 0.1],
            "epochs": [10, 20, 50, 100, 200],
            "batch_size": [8, 16, 32, 64, 128],
        }

        super().__init__("LSTM", parameters)

        if params is not None:
            width = params.get("width", width)
            depth = params.get("depth", depth)
            dropout_rate = params.get("dropout_rate", dropout_rate)
            learning_rate = params.get("learning_rate", learning_rate)
            epochs = params.get("epochs", epochs)
            batch_size = params.get("batch_size", batch_size)
            max_input_dim = params.get("max_input_dim", max_input_dim)
            max_parameter_bytes = params.get("max_parameter_bytes", max_parameter_bytes)

        self.width = int(width)
        self.depth = int(depth)
        self.dropout_rate = float(dropout_rate)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.max_input_dim = None if max_input_dim is None else int(max_input_dim)
        self.max_parameter_bytes = (
            None if max_parameter_bytes is None else int(max_parameter_bytes)
        )
        self.device = get_device()
        self.needs_training = True
        self.needs_alignment = False
        self.preserves_leading_axis = True
        self.model = None
        self.input_dim = None
        self.num_classes = None
        self.is_trained = False
        self.model_state = None
        self.is_multilabel = False

        self._set_random_seeds()

    def _set_random_seeds(self, seed=42):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _shape_description(modality: Modality) -> str:
        data = modality.data
        if isinstance(data, np.ndarray):
            shape = data.shape
        else:
            shapes = [np.asarray(sample).shape for sample in data[:3]]
            shape = f"ragged samples {shapes}"
        return (
            f"id={modality.modality_id}, type={modality.modality_type}, "
            f"shape={shape}"
        )

    @staticmethod
    def _as_sequence_array(modality: Modality) -> np.ndarray:
        try:
            data = np.asarray(modality.data)
        except ValueError:
            data = np.asarray(modality.data, dtype=object)

        if data.dtype != object:
            if data.ndim == 1:
                return data.reshape(-1, 1, 1)
            if data.ndim == 2:
                return data.reshape(data.shape[0], 1, data.shape[1])
            if data.ndim == 3:
                return data
            raise ValueError(
                "LSTM fusion accepts per-instance vectors or sequences with "
                f"shape (instances, features) or (instances, sequence, features); "
                f"got {LSTM._shape_description(modality)}. Pool spatial dimensions "
                "or apply a feature representation before LSTM fusion."
            )

        samples = []
        feature_dim = None
        for index, sample in enumerate(modality.data):
            sample = np.asarray(sample)
            if sample.ndim == 1:
                sample = sample.reshape(1, -1)
            elif sample.ndim != 2:
                raise ValueError(
                    "Ragged LSTM inputs must contain vectors or (sequence, features) "
                    f"matrices; sample {index} has shape {sample.shape} in "
                    f"{LSTM._shape_description(modality)}."
                )
            if feature_dim is None:
                feature_dim = sample.shape[1]
            elif sample.shape[1] != feature_dim:
                raise ValueError(
                    "Ragged LSTM sequences must have a fixed feature dimension; "
                    f"expected {feature_dim}, got {sample.shape[1]} for sample {index}."
                )
            samples.append(sample)

        if not samples:
            raise ValueError("LSTM fusion requires at least one instance")

        max_seq_len = max(sample.shape[0] for sample in samples)
        result = np.zeros((len(samples), max_seq_len, feature_dim), dtype=np.float32)
        for index, sample in enumerate(samples):
            result[index, : sample.shape[0]] = sample
        return result

    def _prepare_data(self, modalities: List[Modality]) -> np.ndarray:
        if not modalities:
            raise ValueError("LSTM fusion requires at least one modality")

        processed_modalities = [
            self._as_sequence_array(modality) for modality in modalities
        ]
        num_instances = processed_modalities[0].shape[0]
        if any(data.shape[0] != num_instances for data in processed_modalities[1:]):
            shapes = ", ".join(self._shape_description(m) for m in modalities)
            raise ValueError(f"LSTM inputs have different instance counts: {shapes}")

        input_dim = sum(data.shape[2] for data in processed_modalities)
        if self.max_input_dim is not None and input_dim > self.max_input_dim:
            shapes = ", ".join(self._shape_description(m) for m in modalities)
            raise ValueError(
                f"LSTM input feature dimension {input_dim:,} exceeds "
                f"the configured maximum {self.max_input_dim:,}. Inputs: {shapes}"
            )

        max_seq_len = max(mod.shape[1] for mod in processed_modalities)

        aligned_modalities = []
        for data in processed_modalities:
            if data.shape[1] < max_seq_len:
                pad_width = ((0, 0), (0, max_seq_len - data.shape[1]), (0, 0))
                data = np.pad(data, pad_width, mode="constant", constant_values=0)
            aligned_modalities.append(data)

        concatenated_data = np.concatenate(aligned_modalities, axis=2)
        return concatenated_data.astype(np.float32)

    def _parameter_bytes(self, input_dim: int, num_classes: int) -> int:
        directions = 2
        gates = 4
        parameter_count = 0
        layer_input_dim = input_dim
        for _ in range(self.depth):
            per_direction = (
                gates * self.width * layer_input_dim
                + gates * self.width * self.width
                + 2 * gates * self.width
            )
            parameter_count += directions * per_direction
            layer_input_dim = directions * self.width
        parameter_count += directions * self.width * num_classes + num_classes
        return parameter_count * np.dtype(np.float32).itemsize

    def _validate_model_size(self, input_dim: int, num_classes: int) -> None:
        parameter_bytes = self._parameter_bytes(input_dim, num_classes)
        if (
            self.max_parameter_bytes is not None
            and parameter_bytes > self.max_parameter_bytes
        ):
            raise ValueError(
                f"LSTM parameters require approximately "
                f"{parameter_bytes / 1024**3:.2f} GiB, exceeding the configured "
                f"maximum {self.max_parameter_bytes / 1024**3:.2f} GiB "
                f"(input_dim={input_dim:,}, width={self.width}, depth={self.depth})."
            )

    def get_output_stats(self, input_stats) -> RepresentationStats:
        stats_list = self._as_stats_list(input_stats)
        num_instances = max((stats.num_instances for stats in stats_list), default=0)
        return RepresentationStats(
            num_instances,
            (2 * self.width,),
            output_shape_is_known=all(
                stats.output_shape_is_known for stats in stats_list
            ),
            dtype=np.float32,
        )

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        stats_list = self._as_stats_list(input_stats)
        input_dim = 0
        max_seq_len = 1
        for stats in stats_list:
            shape = tuple(stats.output_shape)
            if len(shape) == 1:
                input_dim += shape[0]
            elif len(shape) == 2:
                max_seq_len = max(max_seq_len, shape[0])
                input_dim += shape[1]
            else:
                return {"cpu_peak_bytes": 0, "gpu_peak_bytes": 2**63 - 1}
        parameter_bytes = self._parameter_bytes(input_dim, 2)
        input_bytes = sum(stats_bytes(stats) for stats in stats_list)
        batch_activations = self.batch_size * max_seq_len * 2 * self.width * 4
        return {
            "cpu_peak_bytes": input_bytes + parameter_bytes,
            "gpu_peak_bytes": 4 * parameter_bytes + 3 * batch_activations,
        }

    def _build_model(self, input_dim: int, num_classes: int) -> nn.Module:

        class LSTMClassifier(nn.Module):
            def __init__(
                self, input_dim, hidden_dim, num_layers, num_classes, dropout_rate
            ):
                super(LSTMClassifier, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers

                self.lstm = nn.LSTM(
                    input_dim,
                    hidden_dim,
                    num_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout_rate if num_layers > 1 else 0,
                )

                self.dropout = nn.Dropout(dropout_rate)
                self.classifier = nn.Linear(hidden_dim * 2, num_classes)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_output = lstm_out[:, -1, :]
                dropped = self.dropout(last_output)
                output = self.classifier(dropped)

                return last_output, output

        return LSTMClassifier(
            input_dim, self.width, self.depth, num_classes, self.dropout_rate
        )

    def execute(self, modalities: List[Modality], labels: np.ndarray = None):
        if labels is None:
            raise ValueError("LSTM fusion requires labels for training")

        X = self._prepare_data(modalities)
        y = np.array(labels)

        if y.ndim == 2 and y.shape[1] > 1:
            self.is_multilabel = True
            self.num_classes = y.shape[1]
        else:
            self.is_multilabel = False
            if y.ndim == 2:
                y = y.ravel()
            self.num_classes = len(np.unique(y))

        self.input_dim = X.shape[2]
        self._validate_model_size(self.input_dim, self.num_classes)

        self.model = self._build_model(self.input_dim, self.num_classes)
        self.model = self.model.to(self.device)

        if self.is_multilabel:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        X_tensor = torch.FloatTensor(X)
        if self.is_multilabel:
            y_tensor = torch.FloatTensor(y)
        else:
            y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()

                features, predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        self.is_trained = True

        self.model_state = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "is_multilabel": self.is_multilabel,
            "width": self.width,
            "depth": self.depth,
            "dropout_rate": self.dropout_rate,
        }

        self.model.eval()
        all_features = []
        with torch.no_grad():
            inference_dataloader = DataLoader(
                TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False
            )
            for (batch_X,) in inference_dataloader:
                batch_X = batch_X.to(self.device)
                features, _ = self.model(batch_X)
                all_features.append(features.cpu())

        return torch.cat(all_features, dim=0).numpy()

    def apply_representation(self, modalities: List[Modality]) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before applying representation")

        X = self._prepare_data(modalities)
        if X.shape[2] != self.input_dim:
            raise ValueError(
                f"LSTM was fitted with input_dim={self.input_dim:,}, but received "
                f"input_dim={X.shape[2]:,}."
            )

        self.model = self.model.to(self.device)

        X_tensor = torch.FloatTensor(X)
        all_features = []
        self.model.eval()
        with torch.no_grad():
            inference_dataloader = DataLoader(
                TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False
            )
            for (batch_X,) in inference_dataloader:
                batch_X = batch_X.to(self.device)
                features, _ = self.model(batch_X)
                all_features.append(features.cpu())

        return torch.cat(all_features, dim=0).numpy()

    def get_model_state(self) -> Dict[str, Any]:
        return self.model_state

    def set_model_state(self, state: Dict[str, Any]):
        self.model_state = state
        self.input_dim = state["input_dim"]
        self.num_classes = state["num_classes"]
        self.is_multilabel = state.get("is_multilabel", False)
        self.width = int(state.get("width", self.width))
        self.depth = int(state.get("depth", self.depth))
        self.dropout_rate = float(state.get("dropout_rate", self.dropout_rate))

        self._validate_model_size(self.input_dim, self.num_classes)
        self.model = self._build_model(self.input_dim, self.num_classes)
        self.model.load_state_dict(state["state_dict"])
        self.is_trained = True

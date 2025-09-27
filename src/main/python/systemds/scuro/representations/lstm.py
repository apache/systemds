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

from systemds.scuro.drsearch.operator_registry import register_fusion_operator


@register_fusion_operator()
class LSTM(Fusion):
    def __init__(
        self,
        width=128,
        depth=1,
        dropout_rate=0.1,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
    ):
        parameters = {
            "width": [128, 256, 512],
            "depth": [1, 2, 3],
            "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
            "learning_rate": [0.001, 0.0001, 0.01, 0.1],
            "epochs": [50, 100, 200],
            "batch_size": [8, 16, 32, 64, 128],
        }

        super().__init__("LSTM", parameters)

        self.width = int(width)
        self.depth = int(depth)
        self.dropout_rate = float(dropout_rate)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)

        self.needs_training = True
        self.needs_alignment = True
        self.model = None
        self.input_dim = None
        self.num_classes = None
        self.is_trained = False
        self.model_state = None

        self._set_random_seeds()

    def _set_random_seeds(self, seed=42):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _prepare_data(self, modalities: List[Modality]) -> np.ndarray:
        processed_modalities = []

        for modality in modalities:
            data = np.array(modality.data)

            if data.ndim == 1:
                data = data.reshape(-1, 1, 1)
            elif data.ndim == 2:
                data = data.reshape(data.shape[0], 1, data.shape[1])
            elif data.ndim == 3:
                pass
            else:
                raise ValueError(
                    f"Unsupported data shape: {data.shape}. Expected 1D, 2D, or 3D arrays."
                )

            processed_modalities.append(data)

        max_seq_len = max(mod.shape[1] for mod in processed_modalities)

        aligned_modalities = []
        for data in processed_modalities:
            if data.shape[1] < max_seq_len:
                pad_width = ((0, 0), (0, max_seq_len - data.shape[1]), (0, 0))
                data = np.pad(data, pad_width, mode="constant", constant_values=0)
            aligned_modalities.append(data)

        concatenated_data = np.concatenate(aligned_modalities, axis=2)

        return concatenated_data.astype(np.float32)

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

        self.input_dim = X.shape[2]
        self.num_classes = len(np.unique(y))

        self.model = self._build_model(self.input_dim, self.num_classes)
        device = get_device()
        self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
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
            "width": self.width,
            "depth": self.depth,
            "dropout_rate": self.dropout_rate,
        }

    def apply_representation(self, modalities: List[Modality]) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before applying representation")

        X = self._prepare_data(modalities)

        device = get_device()
        self.model.to(device)

        X_tensor = torch.FloatTensor(X).to(device)

        self.model.eval()
        with torch.no_grad():
            features, _ = self.model(X_tensor)

        return features.cpu().numpy()

    def get_model_state(self) -> Dict[str, Any]:
        return self.model_state

    def set_model_state(self, state: Dict[str, Any]):
        self.model_state = state
        self.input_dim = state["input_dim"]
        self.num_classes = state["num_classes"]

        self.model = self._build_model(self.input_dim, self.num_classes)
        self.model.load_state_dict(state["state_dict"])
        self.is_trained = True

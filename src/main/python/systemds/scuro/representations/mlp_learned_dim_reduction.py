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
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
from systemds.scuro.utils.static_variables import get_device

from systemds.scuro.drsearch.operator_registry import (
    register_dimensionality_reduction_operator,
)
from systemds.scuro.representations.dimensionality_reduction import (
    DimensionalityReduction,
)
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.utils.utils import set_random_seeds


# @register_dimensionality_reduction_operator(ModalityType.EMBEDDING)
class MLPLearnedDimReduction(DimensionalityReduction):
    """
    Learned dimensionality reduction using MLP
    This operator is used to reduce the dimensionality of a representation using a learned MLP.
    Parameters:
    :param output_dim: The number of dimensions to reduce the representation to
    :param batch_size: The batch size to use for training
    :param learning_rate: The learning rate to use for training
    :param epochs: The number of epochs to train for
    """

    def __init__(self, output_dim=256, batch_size=32, learning_rate=0.001, epochs=5):
        parameters = {
            "output_dim": [64, 128, 256, 512, 1024],
            "batch_size": [8, 16, 32, 64, 128],
            "learning_rate": [0.001, 0.0001, 0.01, 0.1],
            "epochs": [5, 10, 20, 50, 100],
        }
        super().__init__("MLPLearnedDimReduction", parameters)
        self.output_dim = output_dim
        self.needs_training = True
        set_random_seeds()
        self.is_multilabel = False
        self.num_classes = 0
        self.is_trained = False
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None

    def execute_with_training(self, data, labels):
        if labels is None:
            raise ValueError("MLP labels requires labels for training")

        X = np.array(data)
        y = np.array(labels)

        if y.ndim == 2 and y.shape[1] > 1:
            self.is_multilabel = True
            self.num_classes = y.shape[1]
        else:
            self.is_multilabel = False
            if y.ndim == 2:
                y = y.ravel()
            self.num_classes = len(np.unique(y))

        input_dim = X.shape[1]
        device = get_device()
        self.model = None
        self.is_trained = False

        self.model = self._build_model(input_dim, self.output_dim, self.num_classes).to(
            device
        )
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
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()

                features, predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        self.is_trained = True
        self.model.eval()
        all_features = []
        with torch.no_grad():
            inference_dataloader = DataLoader(
                TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False
            )
            for (batch_X,) in inference_dataloader:
                batch_X = batch_X.to(device)
                features, _ = self.model(batch_X)
                all_features.append(features.cpu())

        return torch.cat(all_features, dim=0).numpy()

    def apply_representation(self, data) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before applying representation")

        device = get_device()
        self.model.to(device)
        X = np.array(data)
        X_tensor = torch.FloatTensor(X)
        all_features = []
        self.model.eval()
        with torch.no_grad():
            inference_dataloader = DataLoader(
                TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False
            )
            for (batch_X,) in inference_dataloader:
                batch_X = batch_X.to(device)
                features, _ = self.model(batch_X)
                all_features.append(features.cpu())

        return torch.cat(all_features, dim=0).numpy()

    def _build_model(self, input_dim, output_dim, num_classes):

        class MLP(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(MLP, self).__init__()
                self.layers = nn.Sequential(nn.Linear(input_dim, output_dim))

                self.classifier = nn.Linear(output_dim, num_classes)

            def forward(self, x):
                output = self.layers(x)
                return output, self.classifier(output)

        return MLP(input_dim, output_dim)

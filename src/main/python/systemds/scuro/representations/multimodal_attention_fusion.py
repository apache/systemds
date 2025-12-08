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
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any
import numpy as np
from systemds.scuro.drsearch.operator_registry import register_fusion_operator
from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.utils.static_variables import get_device


@register_fusion_operator()
class AttentionFusion(Fusion):
    def __init__(
        self,
        hidden_dim=256,
        num_heads=8,
        dropout=0.1,
        batch_size=32,
        num_epochs=20,
        learning_rate=0.001,
    ):
        parameters = {
            "hidden_dim": [32, 128, 256, 384, 512, 768],
            "num_heads": [2, 4, 8, 12],
            "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
            "batch_size": [8, 16, 32, 64, 128],
            "num_epochs": [10, 20, 50, 100, 150, 200],
            "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2],
        }
        super().__init__("AttentionFusion", parameters)

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.learning_rate = float(learning_rate)

        self.needs_training = True
        self.needs_alignment = True
        self.encoder = None
        self.classification_head = None
        self.input_dim = None
        self.max_sequence_length = None
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

    def _prepare_data(self, modalities: List[Modality]):
        inputs = {}
        input_dimensions = {}
        max_sequence_length = 0

        for i, modality in enumerate(modalities):
            modality_name = f"modality_{i}"
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

            input_dimensions[modality_name] = data.shape[2]  # Feature dimension
            max_sequence_length = max(max_sequence_length, data.shape[1])

            inputs[modality_name] = torch.from_numpy(data.astype(np.float32))

        for modality_name, tensor in inputs.items():
            if tensor.shape[1] < max_sequence_length:
                pad_width = (0, 0, 0, max_sequence_length - tensor.shape[1], 0, 0)
                inputs[modality_name] = F.pad(
                    tensor, pad_width, mode="constant", value=0
                )

        return inputs, input_dimensions, max_sequence_length

    def execute(self, modalities: List[Modality], labels: np.ndarray = None):
        if labels is None:
            raise ValueError("Attention fusion requires labels for training")

        inputs, input_dimensions, max_sequence_length = self._prepare_data(modalities)
        y = np.array(labels)

        if y.ndim == 2 and y.shape[1] > 1:
            self.is_multilabel = True
            self.num_classes = y.shape[1]
        else:
            self.is_multilabel = False
            if y.ndim == 2:
                y = y.ravel()
            self.num_classes = len(np.unique(y))

        self.input_dim = input_dimensions
        self.max_sequence_length = max_sequence_length

        self.encoder = MultiModalAttentionFusion(
            self.input_dim,
            self.hidden_dim,
            self.num_heads,
            self.dropout,
            self.max_sequence_length,
        )

        self.classification_head = FusedClassificationHead(
            fused_dim=self.hidden_dim, num_classes=self.num_classes
        )

        device = get_device()
        self.encoder.to(device)
        self.classification_head.to(device)

        if self.is_multilabel:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.classification_head.parameters()),
            lr=self.learning_rate,
        )

        for modality_name in inputs:
            inputs[modality_name] = inputs[modality_name].to(device)

        if self.is_multilabel:
            labels_tensor = torch.from_numpy(y).float().to(device)
        else:
            labels_tensor = torch.from_numpy(y).long().to(device)

        dataset_inputs = []
        for i in range(len(y)):
            sample_inputs = {name: tensor[i] for name, tensor in inputs.items()}
            dataset_inputs.append((sample_inputs, labels_tensor[i]))

        self.encoder.train()
        self.classification_head.train()

        for epoch in range(self.num_epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch_start in range(0, len(dataset_inputs), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(dataset_inputs))
                batch_data = dataset_inputs[batch_start:batch_end]

                batch_inputs = {}
                batch_labels = []

                for sample_inputs, label in batch_data:
                    batch_labels.append(label)
                    for modality_name, tensor in sample_inputs.items():
                        if modality_name not in batch_inputs:
                            batch_inputs[modality_name] = []
                        batch_inputs[modality_name].append(tensor)

                for modality_name in batch_inputs:
                    batch_inputs[modality_name] = torch.stack(
                        batch_inputs[modality_name]
                    )

                batch_labels = torch.stack(batch_labels)

                optimizer.zero_grad()

                encoder_output = self.encoder(batch_inputs)
                logits = self.classification_head(encoder_output["fused"])
                loss = criterion(logits, batch_labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if self.is_multilabel:
                    predicted = (torch.sigmoid(logits) > 0.5).float()
                    correct = (predicted == batch_labels).float()
                    hamming_acc = correct.mean()
                    total_correct += hamming_acc.item() * batch_labels.size(0)
                    total_samples += batch_labels.size(0)
                else:
                    _, predicted = torch.max(logits.data, 1)
                    total_correct += (predicted == batch_labels).sum().item()
                    total_samples += batch_labels.size(0)

        self.is_trained = True

        self.model_state = {
            "encoder_state_dict": self.encoder.state_dict(),
            "classification_head_state_dict": self.classification_head.state_dict(),
            "input_dimensions": self.input_dim,
            "max_sequence_length": self.max_sequence_length,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        }

        all_features = []

        with torch.no_grad():
            for batch_start in range(
                0, len(inputs[list(inputs.keys())[0]]), self.batch_size
            ):
                batch_end = min(
                    batch_start + self.batch_size, len(inputs[list(inputs.keys())[0]])
                )

                batch_inputs = {}
                for modality_name, tensor in inputs.items():
                    batch_inputs[modality_name] = tensor[batch_start:batch_end]

                encoder_output = self.encoder(batch_inputs)
                all_features.append(encoder_output["fused"].cpu())

        return torch.cat(all_features, dim=0).numpy()

    def apply_representation(self, modalities: List[Modality]) -> np.ndarray:
        if not self.is_trained or self.encoder is None:
            raise ValueError("Model must be trained before applying representation")

        inputs, _, _ = self._prepare_data(modalities)

        device = get_device()
        self.encoder.to(device)

        for modality_name in inputs:
            inputs[modality_name] = inputs[modality_name].to(device)

        self.encoder.eval()
        all_features = []

        with torch.no_grad():
            batch_size = self.batch_size
            n_samples = len(inputs[list(inputs.keys())[0]])

            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)

                batch_inputs = {}
                for modality_name, tensor in inputs.items():
                    batch_inputs[modality_name] = tensor[batch_start:batch_end]

                encoder_output = self.encoder(batch_inputs)
                all_features.append(encoder_output["fused"].cpu())

        return torch.cat(all_features, dim=0).numpy()

    def get_model_state(self) -> Dict[str, Any]:
        return self.model_state

    def set_model_state(self, state: Dict[str, Any]):
        self.model_state = state
        self.input_dim = state["input_dimensions"]
        self.max_sequence_length = state["max_sequence_length"]
        self.num_classes = state["num_classes"]
        self.is_multilabel = state.get("is_multilabel", False)

        self.encoder = MultiModalAttentionFusion(
            self.input_dim,
            state["hidden_dim"],
            state["num_heads"],
            state["dropout"],
            self.max_sequence_length,
        )
        self.encoder.load_state_dict(state["encoder_state_dict"])

        self.classification_head = FusedClassificationHead(
            fused_dim=state["hidden_dim"], num_classes=self.num_classes
        )
        self.classification_head.load_state_dict(
            state["classification_head_state_dict"]
        )

        self.is_trained = True


class FusedClassificationHead(nn.Module):

    def __init__(self, fused_dim, num_classes=2):
        super(FusedClassificationHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(self, fused):
        return self.head(fused)


class MultiModalAttentionFusion(nn.Module):
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        max_seq_len: int,
        pooling_strategy: str = "mean",
    ):
        super().__init__()

        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pooling_strategy = pooling_strategy
        self.max_seq_len = max_seq_len

        self.modality_projections = nn.ModuleDict(
            {
                modality: nn.Linear(dim, hidden_dim)
                for modality, dim in modality_dims.items()
            }
        )

        self.positional_encoding = nn.Parameter(
            torch.randn(max_seq_len, hidden_dim) * 0.1
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        if pooling_strategy == "attention":
            self.pooling_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

        self.modality_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.final_projection = nn.Linear(hidden_dim, hidden_dim)

    def _handle_input_format(self, modality_tensor: torch.Tensor) -> torch.Tensor:
        if len(modality_tensor.shape) == 2:
            modality_tensor = modality_tensor.unsqueeze(1)
        elif len(modality_tensor.shape) == 3:
            pass
        else:
            raise ValueError(
                f"Input tensor must be 2D or 3D, got {len(modality_tensor.shape)}D"
            )

        if modality_tensor.dtype != torch.float:
            modality_tensor = modality_tensor.float()

        return modality_tensor

    def _pool_sequence(
        self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.pooling_strategy == "mean":
            if mask is not None:
                # Masked mean pooling
                masked_seq = sequence * mask.unsqueeze(-1)
                return masked_seq.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(
                    min=1
                )
            else:
                return sequence.mean(dim=1)

        elif self.pooling_strategy == "max":
            if mask is not None:
                masked_seq = sequence.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                return masked_seq.max(dim=1)[0]
            else:
                return sequence.max(dim=1)[0]

        elif self.pooling_strategy == "cls":
            return sequence[:, 0, :]

        elif self.pooling_strategy == "attention":
            attention_scores = self.pooling_attention(sequence).squeeze(-1)

            if mask is not None:
                attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

            attention_weights = F.softmax(attention_scores, dim=1)
            return (sequence * attention_weights.unsqueeze(-1)).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        modality_embeddings = {}

        for modality, input_tensor in modality_inputs.items():
            normalized_input = self._handle_input_format(input_tensor)
            seq_len = normalized_input.size(1)

            projected = self.modality_projections[modality](normalized_input)

            if seq_len > 1:
                pos_encoding = self.positional_encoding[:seq_len].unsqueeze(0)
                projected = projected + pos_encoding

            if seq_len > 1:
                mask = modality_masks.get(modality) if modality_masks else None

                attended, _ = self.self_attention(
                    query=projected,
                    key=projected,
                    value=projected,
                    key_padding_mask=~mask if mask is not None else None,
                )

                projected = self.layer_norm(projected + self.dropout_layer(attended))

                pooled = self._pool_sequence(projected, mask)
            else:
                pooled = projected.squeeze(1)

            modality_embeddings[modality] = pooled

        if len(modality_embeddings) > 1:
            modality_stack = torch.stack(list(modality_embeddings.values()), dim=1)

            cross_attended, cross_attention_weights = self.cross_attention(
                query=modality_stack, key=modality_stack, value=modality_stack
            )

            cross_attended = self.layer_norm(
                modality_stack + self.dropout_layer(cross_attended)
            )

            updated_embeddings = {
                modality: cross_attended[:, i, :]
                for i, modality in enumerate(modality_embeddings.keys())
            }
            modality_embeddings = updated_embeddings

        modality_stack = torch.stack(list(modality_embeddings.values()), dim=1)
        modality_scores = self.modality_attention(modality_stack).squeeze(-1)
        modality_weights = F.softmax(modality_scores, dim=1)

        fused_representation = (modality_stack * modality_weights.unsqueeze(-1)).sum(
            dim=1
        )

        output = self.final_projection(fused_representation)

        return {
            "fused": output,
            "modality_embeddings": modality_embeddings,
            "attention_weights": modality_weights,
        }

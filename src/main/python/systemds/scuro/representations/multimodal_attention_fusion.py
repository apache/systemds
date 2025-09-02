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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
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
        fusion_strategy="attention",
        batch_size=32,
        num_epochs=50,
    ):
        self.encoder = None
        params = {
            "hidden_dim": [128, 256, 512],
            "num_heads": [1, 4, 8],
            "dropout": [0.1, 0.2, 0.3],
            "fusion_strategy": ["mean", "max", "attention", "cls"],
            "batch_size": [32, 64, 128],
            "num_epochs": [50, 70, 100, 150],
        }
        super().__init__("AttentionFusion", params)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.fusion_strategy = fusion_strategy
        self.batch_size = batch_size
        self.needs_training = True
        self.needs_instance_alignment = True
        self.num_epochs = num_epochs

    def execute(
        self,
        data: List[np.ndarray],
        labels: np.ndarray,
    ):
        input_dimension = {}
        inputs = {}
        max_sequence_length = 0
        masks = {}
        for i, modality in enumerate(data):
            modality_name = "modality_" + str(i)
            shape = modality.shape
            max_sequence_length = max(max_sequence_length, shape[1])
            input_dimension[modality_name] = shape[2] if len(shape) > 2 else shape[1]
            inputs[modality_name] = torch.from_numpy(np.stack(modality)).to(
                get_device()
            )

            # attention_masks_list = [
            #     entry["attention_masks"]
            #     for entry in modality.metadata.values()
            #     if "attention_masks" in entry
            # ]
            attention_masks_list = None
            if attention_masks_list:
                masks[modality_name] = (
                    torch.tensor(np.array(attention_masks_list)).bool().to(get_device())
                )
            else:
                masks[modality_name] = None

        self.encoder = MultiModalAttentionFusion(
            input_dimension,
            self.hidden_dim,
            self.num_heads,
            self.dropout,
            max_sequence_length,
            self.fusion_strategy,
        )

        head = FusedClassificationHead(
            fused_dim=self.hidden_dim, num_classes=len(np.unique(labels))
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(head.parameters()), lr=0.001
        )
        labels = torch.from_numpy(labels).to(get_device())

        for epoch in range(self.num_epochs):
            total_loss = 0
            total_accuracy = 0
            for batch_idx in range(0, len(data), self.batch_size):
                batched_input = {}
                for modality, modality_data in inputs.items():
                    batched_input[modality] = modality_data[
                        batch_idx : batch_idx + self.batch_size
                    ]
                loss, predictions = self.train_encoder_step(
                    head,
                    inputs,
                    labels[batch_idx : batch_idx + self.batch_size],
                    criterion,
                    optimizer,
                )
                total_loss += loss
                total_accuracy += predictions

            if epoch % 50 == 0 or epoch == self.num_epochs - 1:
                print(
                    f"Epoch {epoch}, Loss: {total_loss:.4f}, accuracy: {total_accuracy/len(data):.4f}"
                )

    # Training step (encoder + classification head)
    def train_encoder_step(self, head, inputs, labels, criterion, optimizer):
        self.encoder.train()
        head.train()
        optimizer.zero_grad()
        output = self.encoder(inputs)
        logits = head(output["fused"])
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(logits.data, 1)
        return loss.item(), (predicted == labels).sum().item()

    def apply_representation(self, modalities):
        inputs = {}
        for i, modality in enumerate(modalities):
            modality_name = "modality_" + str(i)
            inputs[modality_name] = torch.from_numpy(np.stack(modality)).to(
                get_device()
            )
        self.encoder.eval()
        with torch.no_grad():
            output = self.encoder(inputs)
        return output["fused"].cpu().numpy()


class FusedClassificationHead(nn.Module):
    """
    Simple classification head for supervision during training.
    """

    def __init__(self, fused_dim, num_classes=2):
        super(FusedClassificationHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Linear(fused_dim // 2, num_classes),
        ).to(get_device())

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
        pooling_strategy: str,
    ):
        super().__init__()

        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pooling_strategy = pooling_strategy
        self.max_seq_len = max_seq_len

        # Project each modality to the same hidden dimension
        self.modality_projections = nn.ModuleDict(
            {
                modality: nn.Linear(dim, hidden_dim).to(get_device())
                for modality, dim in modality_dims.items()
            }
        )

        # Positional encoding for sequence modalities
        self.positional_encoding = nn.Parameter(
            torch.randn(max_seq_len, hidden_dim) * 0.1
        ).to(get_device())

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        ).to(get_device())

        # Self-attention within each modality
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        ).to(get_device())

        # Attention-based pooling for sequences
        if pooling_strategy == "attention":
            self.pooling_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            ).to(get_device())

        # Modality-level attention for final fusion
        self.modality_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        ).to(get_device())

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim).to(get_device())
        self.dropout = nn.Dropout(dropout).to(get_device())

        # Final projection
        self.final_projection = nn.Linear(hidden_dim, hidden_dim).to(get_device())

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
                # Set masked positions to large negative value before max pooling
                masked_seq = sequence.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                return masked_seq.max(dim=1)[0]
            else:
                return sequence.max(dim=1)[0]

        elif self.pooling_strategy == "cls":
            # Use the first token (assuming it's a CLS token)
            return sequence[:, 0, :]

        elif self.pooling_strategy == "attention":
            # Attention-based pooling
            attention_scores = self.pooling_attention(sequence).squeeze(
                -1
            )  # (batch, seq)

            if mask is not None:
                attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

            attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq)
            return (sequence * attention_weights.unsqueeze(-1)).sum(
                dim=1
            )  # (batch, hidden)

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

                projected = self.layer_norm(projected + self.dropout(attended))

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
                modality_stack + self.dropout(cross_attended)
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

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
from systemds.scuro.utils.torch_dataset import CustomDataset
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.drsearch.operator_registry import register_representation
import torch.utils.data
import torch
import numpy as np
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.utils.static_variables import get_device
from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TextDataset(Dataset):
    def __init__(self, texts):

        self.texts = []
        if isinstance(texts, list):
            self.texts = texts
        else:
            for text in texts:
                if text is None:
                    self.texts.append("")
                elif isinstance(text, np.ndarray):
                    self.texts.append(str(text.item()) if text.size == 1 else str(text))
                elif not isinstance(text, str):
                    self.texts.append(str(text))
                else:
                    self.texts.append(text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


# @register_representation([ModalityType.TEXT])
class ELMoRepresentation(UnimodalRepresentation):
    def __init__(
        self, model_name="elmo-original", layer="mix", pooling="mean", output_file=None
    ):
        self.data_type = torch.float32
        self.model_name = model_name
        self.layer_name = layer
        self.pooling = pooling  # "mean", "max", "first", "last", or "all" (no pooling)
        parameters = self._get_parameters()
        super().__init__("ELMo", ModalityType.EMBEDDING, parameters)

        self.output_file = output_file

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name

        if model_name == "elmo-original":
            self.model = ELMoEmbeddings("original")
            self.embedding_dim = 1024
        elif model_name == "elmo-small":
            self.model = ELMoEmbeddings("small")
            self.embedding_dim = 256
        elif model_name == "elmo-medium":
            self.model = ELMoEmbeddings("medium")
            self.embedding_dim = 512
        else:
            raise NotImplementedError(f"Model {model_name} not supported")

        self.model = self.model.to(get_device())

    def _get_parameters(self):
        parameters = {
            "model_name": ["elmo-original", "elmo-small", "elmo-medium"],
            "layer_name": [
                "mix",
                "layer_0",
                "layer_1",
                "layer_2",
            ],
            "pooling": ["mean", "max", "first", "last", "all"],
        }
        return parameters

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality, self, ModalityType.EMBEDDING
        )
        dataset = TextDataset(modality.data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=None)
        embeddings = []
        for batch in dataloader:
            texts = batch
            for text in texts:
                sentence = Sentence(text)
                self.model.embed(sentence)
                token_embeddings = []
                for token in sentence:
                    if self.layer_name == "mix":
                        embedding = token.embedding
                    elif self.layer_name == "layer_0":
                        embedding = token.get_embedding(self.model.name + "-0")
                    elif self.layer_name == "layer_1":
                        embedding = token.get_embedding(self.model.name + "-1")
                    elif self.layer_name == "layer_2":
                        embedding = token.get_embedding(self.model.name + "-2")
                    else:
                        embedding = token.embedding

                    token_embeddings.append(embedding.cpu().numpy())

                token_embeddings = np.array(token_embeddings)

                if self.pooling == "mean":
                    sentence_embedding = np.mean(token_embeddings, axis=0)
                elif self.pooling == "max":
                    sentence_embedding = np.max(token_embeddings, axis=0)
                elif self.pooling == "first":
                    sentence_embedding = token_embeddings[0]
                elif self.pooling == "last":
                    sentence_embedding = token_embeddings[-1]
                elif self.pooling == "all":
                    sentence_embedding = token_embeddings.flatten()
                else:
                    sentence_embedding = np.mean(token_embeddings, axis=0)

                embeddings.append(sentence_embedding.astype(np.float32))

        transformed_modality.data = np.array(embeddings)
        return transformed_modality

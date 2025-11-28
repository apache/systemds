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
import numpy as np
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
import torch
from transformers import BertTokenizerFast, BertModel
from systemds.scuro.representations.utils import save_embeddings
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation
from systemds.scuro.utils.static_variables import get_device
import os
from torch.utils.data import Dataset, DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextDataset(Dataset):
    def __init__(self, texts):

        self.texts = []
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


@register_representation(ModalityType.TEXT)
class Bert(UnimodalRepresentation):
    def __init__(self, model_name="bert", output_file=None, max_seq_length=512):
        parameters = {"model_name": "bert"}
        self.model_name = model_name
        super().__init__("Bert", ModalityType.EMBEDDING, parameters)

        self.output_file = output_file
        self.max_seq_length = max_seq_length

    def transform(self, modality):
        transformed_modality = TransformedModality(modality, self)
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(
            model_name, clean_up_tokenization_spaces=True
        )

        model = BertModel.from_pretrained(model_name).to(get_device())

        embeddings = self.create_embeddings(modality, model, tokenizer)

        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality.data_type = np.float32
        transformed_modality.data = np.array(embeddings)
        return transformed_modality

    def create_embeddings(self, modality, model, tokenizer):
        dataset = TextDataset(modality.data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=None)
        cls_embeddings = []
        for batch in dataloader:
            inputs = tokenizer(
                batch,
                return_offsets_mapping=True,
                return_tensors="pt",
                padding="max_length",
                return_attention_mask=True,
                truncation=True,
                max_length=512,  # TODO: make this dynamic
            )

            inputs.to(get_device())
            ModalityType.TEXT.add_field_for_instances(
                modality.metadata,
                "token_to_character_mapping",
                inputs.data["offset_mapping"].tolist(),
            )

            ModalityType.TEXT.add_field_for_instances(
                modality.metadata,
                "attention_masks",
                inputs.data["attention_mask"].tolist(),
            )
            del inputs.data["offset_mapping"]

            with torch.no_grad():
                outputs = model(**inputs)

                cls_embedding = outputs.last_hidden_state.detach().cpu().numpy()
                cls_embeddings.extend(cls_embedding)

        return np.array(cls_embeddings)

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
from transformers import AutoTokenizer, AutoModel
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


class BertFamily(UnimodalRepresentation):
    def __init__(
        self,
        representation_name,
        model_name,
        layer,
        parameters={},
        output_file=None,
        max_seq_length=512,
    ):
        self.model_name = model_name
        super().__init__(representation_name, ModalityType.EMBEDDING, parameters)

        self.layer_name = layer
        self.output_file = output_file
        self.max_seq_length = max_seq_length
        self.needs_context = True
        self.initial_context_length = 350

    def transform(self, modality):
        transformed_modality = TransformedModality(modality, self)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, clean_up_tokenization_spaces=True
        )
        self.model = AutoModel.from_pretrained(self.model_name).to(get_device())
        self.bert_output = None

        def get_activation(name):
            def hook(model, input, output):
                self.bert_output = output.detach().cpu().numpy()

            return hook

        if self.layer_name != "cls":
            for name, layer in self.model.named_modules():
                if name == self.layer_name:
                    layer.register_forward_hook(get_activation(name))
                    break

        if isinstance(modality.data[0], list):
            embeddings = []
            for d in modality.data:
                embeddings.append(self.create_embeddings(d, self.model, tokenizer))
        else:
            embeddings = self.create_embeddings(modality.data, self.model, tokenizer)

        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality.data_type = np.float32
        transformed_modality.data = embeddings
        return transformed_modality

    def create_embeddings(self, data, model, tokenizer):
        dataset = TextDataset(data)
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
                max_length=512,  # TODO: make this dynamic with parameter to tune
            )

            inputs.to(get_device())
            # ModalityType.TEXT.add_field_for_instances(
            #     modality.metadata,
            #     "token_to_character_mapping",
            #     inputs.data["offset_mapping"].tolist(),
            # )
            #
            # ModalityType.TEXT.add_field_for_instances(
            #     modality.metadata,
            #     "attention_masks",
            #     inputs.data["attention_mask"].tolist(),
            # )
            del inputs.data["offset_mapping"]

            with torch.no_grad():
                outputs = model(**inputs)
                if self.layer_name == "cls":
                    cls_embedding = outputs.last_hidden_state.detach().cpu().numpy()
                else:
                    cls_embedding = self.bert_output
                cls_embeddings.extend(cls_embedding)

        return np.array(cls_embeddings)


@register_representation(ModalityType.TEXT)
class Bert(BertFamily):
    def __init__(self, layer="cls", output_file=None, max_seq_length=512):
        parameters = {
            "layer_name": [
                "cls",
                "encoder.layer.0",
                "encoder.layer.1",
                "encoder.layer.2",
                "encoder.layer.3",
                "encoder.layer.4",
                "encoder.layer.5",
                "encoder.layer.6",
                "encoder.layer.7",
                "encoder.layer.8",
                "encoder.layer.9",
                "encoder.layer.10",
                "encoder.layer.11",
                "pooler",
                "pooler.activation",
            ]
        }
        super().__init__(
            "Bert", "bert-base-uncased", layer, parameters, output_file, max_seq_length
        )


@register_representation(ModalityType.TEXT)
class RoBERTa(BertFamily):
    def __init__(self, layer="cls", output_file=None, max_seq_length=512):
        parameters = {
            "layer_name": [
                "cls",
                "encoder.layer.0",
                "encoder.layer.1",
                "encoder.layer.2",
                "encoder.layer.3",
                "encoder.layer.4",
                "encoder.layer.5",
                "encoder.layer.6",
                "encoder.layer.7",
                "encoder.layer.8",
                "encoder.layer.9",
                "encoder.layer.10",
                "encoder.layer.11",
                "pooler",
                "pooler.activation",
            ]
        }
        super().__init__(
            "RoBERTa", "roberta-base", layer, parameters, output_file, max_seq_length
        )


@register_representation(ModalityType.TEXT)
class DistillBERT(BertFamily):
    def __init__(self, layer="cls", output_file=None, max_seq_length=512):
        parameters = {
            "layer_name": [
                "cls",
                "transformer.layer.0",
                "transformer.layer.1",
                "transformer.layer.2",
                "transformer.layer.3",
                "transformer.layer.4",
                "transformer.layer.5",
            ]
        }
        super().__init__(
            "DistillBERT",
            "distilbert-base-uncased",
            layer,
            parameters,
            output_file,
            max_seq_length,
        )


@register_representation(ModalityType.TEXT)
class ALBERT(BertFamily):
    def __init__(self, layer="cls", output_file=None, max_seq_length=512):
        parameters = {"layer_name": ["cls", "encoder.albert_layer_groups.0", "pooler"]}
        super().__init__(
            "ALBERT", "albert-base-v2", layer, parameters, output_file, max_seq_length
        )


@register_representation(ModalityType.TEXT)
class ELECTRA(BertFamily):
    def __init__(self, layer="cls", output_file=None, max_seq_length=512):
        parameters = {
            "layer_name": [
                "cls",
                "encoder.layer.0",
                "encoder.layer.1",
                "encoder.layer.2",
                "encoder.layer.3",
                "encoder.layer.4",
                "encoder.layer.5",
                "encoder.layer.6",
                "encoder.layer.7",
                "encoder.layer.8",
                "encoder.layer.9",
                "encoder.layer.10",
                "encoder.layer.11",
            ]
        }
        super().__init__(
            "ELECTRA",
            "google/electra-base-discriminator",
            layer,
            parameters,
            output_file,
            max_seq_length,
        )

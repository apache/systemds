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

from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
import torch
from transformers import BertTokenizerFast, BertModel
from systemds.scuro.representations.utils import save_embeddings
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@register_representation(ModalityType.TEXT)
class Bert(UnimodalRepresentation):
    def __init__(self, model_name="bert", output_file=None):
        parameters = {"model_name": "bert"}
        self.model_name = model_name
        super().__init__("Bert", ModalityType.EMBEDDING, parameters)

        self.output_file = output_file

    def transform(self, modality):
        transformed_modality = TransformedModality(modality, self)
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(
            model_name, clean_up_tokenization_spaces=True
        )

        model = BertModel.from_pretrained(model_name)

        embeddings = self.create_embeddings(modality, model, tokenizer)
        
        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality.data = embeddings
        return transformed_modality

    def create_embeddings(self, modality, model, tokenizer):
        embeddings = []
        for i, d in enumerate(modality.data):
            inputs = tokenizer(d, return_offsets_mapping=True, return_tensors="pt", padding=True, truncation=True)
          
            ModalityType.TEXT.add_field(list(modality.metadata.values())[i], "token_to_character_mapping", inputs.data['offset_mapping'][0].tolist())
            
            del inputs.data['offset_mapping']
            
            with torch.no_grad():
                outputs = model(**inputs)

                cls_embedding = outputs.last_hidden_state[0].numpy()
                embeddings.append(cls_embedding)

        return embeddings

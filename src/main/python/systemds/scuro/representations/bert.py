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

import pickle

import numpy as np

from systemds.scuro.representations.unimodal import UnimodalRepresentation
import torch
from transformers import BertTokenizer, BertModel
import os


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


class Bert(UnimodalRepresentation):
    def __init__(self, avg_layers=None, output_file=None):
        super().__init__("Bert")

        self.avg_layers = avg_layers
        self.output_file = output_file

    def parse_all(self, filepath, indices, get_sequences=False):
        # Assumes text is stored in .txt files
        data = []
        if os.path.isdir(filepath):
            for filename in os.listdir(filepath):
                f = os.path.join(filepath, filename)
                if os.path.isfile(f):
                    with open(f, "r") as file:
                        data.append(file.readlines()[0])
        else:
            with open(filepath, "r") as file:
                data = file.readlines()

        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)

        if self.avg_layers is not None:
            model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        else:
            model = BertModel.from_pretrained(model_name)

        embeddings = self.create_embeddings(data, model, tokenizer)

        if self.output_file is not None:
            data = {}
            for i in range(0, embeddings.shape[0]):
                data[indices[i]] = embeddings[i]
            self.save_embeddings(data)

        return embeddings

    def create_embeddings(self, data, model, tokenizer):
        embeddings = []
        for d in data:
            inputs = tokenizer(d, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                outputs = model(**inputs)

            if self.avg_layers is not None:
                cls_embedding = [
                    outputs.hidden_states[i][:, 0, :]
                    for i in range(-self.avg_layers, 0)
                ]
                cls_embedding = torch.mean(torch.stack(cls_embedding), dim=0)
            else:
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)

        embeddings = np.array(embeddings)
        return embeddings.reshape((embeddings.shape[0], embeddings.shape[-1]))

    def save_embeddings(self, data):
        with open(self.output_file, "wb") as file:
            pickle.dump(data, file)

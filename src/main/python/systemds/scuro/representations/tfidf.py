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

from sklearn.feature_extraction.text import TfidfVectorizer
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import save_embeddings

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation


@register_representation(ModalityType.TEXT)
class TfIdf(UnimodalRepresentation):
    def __init__(self, min_df=2, output_file=None):
        parameters = {"min_df": [min_df]}
        super().__init__("TF-IDF", ModalityType.EMBEDDING, parameters)
        self.min_df = min_df
        self.output_file = output_file

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality.modality_type, self, modality.modality_id, modality.metadata
        )

        vectorizer = TfidfVectorizer(min_df=self.min_df)

        X = vectorizer.fit_transform(modality.data)
        X = [np.array(x).reshape(1, -1) for x in X.toarray()]

        if self.output_file is not None:
            save_embeddings(X, self.output_file)

        transformed_modality.data = X
        return transformed_modality

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
from textblob import TextBlob

from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import save_embeddings
from gensim import models
from gensim.corpora import Dictionary

import nltk
nltk.download('punkt_tab')

class TfIdf(UnimodalRepresentation):
    def __init__(self, min_df, output_file=None):
        super().__init__("TF-IDF")
        self.min_df = min_df
        self.output_file = output_file

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality.modality_type, self, modality.metadata
        )

        tokens = [list(TextBlob(s).words) for s in modality.data]
        dictionary = Dictionary()
        BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tokens]
        tfidf = models.TfidfModel(BoW_corpus, smartirs="ntc")
        X = tfidf[BoW_corpus]
        X = [np.array(x)[:, 1].reshape(1, -1) for x in X]

        if self.output_file is not None:
            save_embeddings(X, self.output_file)

        transformed_modality.data = X
        return transformed_modality

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

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import read_data_from_file, save_embeddings


class TfIdf(UnimodalRepresentation):
    def __init__(self, min_df, output_file=None):
        super().__init__("TF-IDF")
        self.min_df = min_df
        self.output_file = output_file

    def parse_all(self, filepath, indices):
        vectorizer = TfidfVectorizer(min_df=self.min_df)

        segments = read_data_from_file(filepath, indices)
        X = vectorizer.fit_transform(segments.values())
        X = X.toarray()

        if self.output_file is not None:
            df = pd.DataFrame(X)
            df.index = segments.keys()

            save_embeddings(df, self.output_file)

        return X

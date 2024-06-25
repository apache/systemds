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
from aligner.alignment import Alignment
from aligner.alignment_strategy import ChunkedCrossCorrelation
from modality.representation import PixelRepresentation
from modality.video_modality import VideoModality
from aligner.similarity_measures import CosineSimilarity

# Setup modalities
file_path_a = ''
file_path_b = ''
representation_a = PixelRepresentation()  # Concrete Representation
representation_b = PixelRepresentation()  # Concrete Representation
modality_a = VideoModality(file_path_a, representation_a)
modality_b = VideoModality(file_path_b, representation_b)

# Align modalities
alignment_strategy = ChunkedCrossCorrelation()  # Concrete Alignment Strategy
similarity_measure = CosineSimilarity()
aligner = Alignment(modality_a, modality_b, alignment_strategy, similarity_measure)
aligned_modality = aligner.align_modalities()

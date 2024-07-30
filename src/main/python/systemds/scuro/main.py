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
import collections
import json
from datetime import datetime

from representations.average import Averaging
from modality.aligned_modality import AlignedModality
from modality.text_modality import TextModality
from modality.video_modality import VideoModality
from modality.audio_modality import AudioModality
from representations.unimodal import Pickle, JSON, HDF5, NPY
from models.discrete_model import DiscreteModel


labels = []
train_indices = []

video_path = ''
audio_path = ''
text_path = ''

# Load modalities (audio, video, text)
video = VideoModality(video_path, HDF5(), train_indices)
audio = AudioModality(audio_path, Pickle(), train_indices)
text = TextModality(text_path, NPY(), train_indices)

video.read_all()
audio.read_all()
text.read_all()

combined_modality = AlignedModality(Averaging(), [text, video, audio])
combined_modality.combine()

# create train-val split
train_X, train_y = None, None
val_X, val_y = None, None

model = DiscreteModel()
model.fit(train_X, train_y)
model.test(val_X, val_y)



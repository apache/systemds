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
from systemds.scuro.modality.aligned_modality import AlignedModality
from systemds.scuro.modality.audio_modality import AudioModality
from systemds.scuro.modality.video_modality import VideoModality
from systemds.scuro.modality.test_modality import TextModality
from systemds.scuro.modality.modality import Modality


__all__ = ["AlignedModality", "AudioModality", "VideoModality", "TextModality", "Modality"]
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
import torch
import imagebind.data as data

from imagebind.models.imagebind_model import ModalityType as IBModalityType

from imagebind.models import imagebind_model
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import save_embeddings

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
# elif torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


@register_representation([ModalityType.TEXT, ModalityType.AUDIO, ModalityType.VIDEO])
class ImageBind(UnimodalRepresentation):
    def __init__(self):
        parameters = {}
        super().__init__("ImageBind", ModalityType.EMBEDDING, parameters)

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality.modality_type, self, modality.modality_id, modality.metadata
        )

        model = imagebind_model.imagebind_huge(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        model.to(DEVICE)

        result = []
        if modality.modality_type == ModalityType.TEXT:
            for i, instance in enumerate(modality.data):
                text_inputs = data.load_and_transform_text(instance, DEVICE)
                text_embeddings = model({IBModalityType.TEXT: text_inputs})[
                    IBModalityType.TEXT
                ]
                result.append(text_embeddings.mean(axis=0).cpu().detach().numpy())
        if modality.modality_type == ModalityType.AUDIO:
            audio_inputs = data.load_and_transform_audio_data(
                list(modality.metadata),
                DEVICE,
            )
            audio_embeddings = model({IBModalityType.AUDIO: audio_inputs})[
                IBModalityType.AUDIO
            ]
            result.append(audio_embeddings.mean(axis=0).cpu().detach().numpy())
        if modality.modality_type == ModalityType.VIDEO:
            video_inputs = data.load_and_transform_video_data(
                list(modality.metadata)[
                    (modality.data_loader.next_chunk - 1)
                    * (modality.data_loader.chunk_size) : (
                        modality.data_loader.next_chunk - 1
                    )
                    * (modality.data_loader.chunk_size)
                    + (modality.data_loader.chunk_size)
                ],
                DEVICE,
            )
            video_embeddings = model({IBModalityType.VISION: video_inputs})[
                IBModalityType.VISION
            ]
            result.append(video_embeddings.mean(axis=0).cpu().detach().numpy())

        transformed_modality.data = result
        return transformed_modality

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
from torchvision import transforms

from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
import torch
from systemds.scuro.representations.utils import save_embeddings
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation
from transformers import CLIPProcessor, CLIPModel

from systemds.scuro.utils.converter import numpy_dtype_to_torch_dtype
from systemds.scuro.utils.static_variables import get_device
from systemds.scuro.utils.torch_dataset import CustomDataset


@register_representation([ModalityType.VIDEO, ModalityType.IMAGE])
class CLIPVisual(UnimodalRepresentation):
    def __init__(self, output_file=None):
        parameters = {}
        super().__init__("CLIPVisual", ModalityType.EMBEDDING, parameters)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            get_device()
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.output_file = output_file

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        self.data_type = torch.float32
        if next(self.model.parameters()).dtype != self.data_type:
            self.model = self.model.to(self.data_type)

        embeddings = self.create_visual_embeddings(modality)

        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality.data = list(embeddings.values())
        return transformed_modality

    def create_visual_embeddings(self, modality):

        clip_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype=self.data_type),
            ]
        )
        dataset = CustomDataset(
            modality.data, self.data_type, get_device(), tf=clip_transform
        )

        embeddings = {}
        for instance in torch.utils.data.DataLoader(dataset):
            id = int(instance["id"][0])
            frames = instance["data"][0]
            embeddings[id] = []
            batch_size = 64

            for start_index in range(0, len(frames), batch_size):
                end_index = min(start_index + batch_size, len(frames))
                frame_ids_range = range(start_index, end_index)
                frame_batch = frames[frame_ids_range]

                inputs = self.processor(images=frame_batch, return_tensors="pt")
                inputs.to(get_device())
                with torch.no_grad():
                    output = self.model.get_image_features(**inputs)

                if len(output.shape) > 2:
                    output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))

                embeddings[id].extend(
                    torch.flatten(output, 1)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .astype(np.float32)
                )

            embeddings[id] = np.array(embeddings[id])
        return embeddings


@register_representation(ModalityType.TEXT)
class CLIPText(UnimodalRepresentation):
    def __init__(self, output_file=None):
        parameters = {}
        super().__init__("CLIPText", ModalityType.EMBEDDING, parameters)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            get_device()
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.output_file = output_file

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )

        embeddings = self.create_text_embeddings(modality.data, self.model)

        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality.data = embeddings
        return transformed_modality

    def create_text_embeddings(self, data, model):
        embeddings = []
        for d in data:
            inputs = self.processor(
                text=d,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs.to(get_device())
            with torch.no_grad():
                text_embedding = model.get_text_features(**inputs)
                embeddings.append(
                    text_embedding.squeeze().detach().cpu().numpy().reshape(1, -1)
                )

        return embeddings

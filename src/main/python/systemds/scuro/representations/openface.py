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
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

from systemds.scuro.dataloader.video_loader import VideoStats
from systemds.scuro.drsearch.operator_registry import (
    register_representation,
    register_expensive_representation,
)
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.representation import (
    CONTAINER_LIST,
    RepresentationStats,
    stats_itemsize,
)
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import get_sequence_lengths
from systemds.scuro.utils.static_variables import (
    NP_ARRAY_HEADER_BYTES,
    PY_LIST_HEADER_BYTES,
    PY_LIST_SLOT_BYTES,
    get_device,
)


@contextmanager
def _without_tf32():
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


_retinaface_pretrain_patched = False
_star_dirs_patched = False


def _patch_openface_package_defaults(needs_landmarks: bool) -> None:
    global _retinaface_pretrain_patched, _star_dirs_patched

    if not _retinaface_pretrain_patched:
        try:
            from openface.Pytorch_Retinaface.data.config import cfg_mnet, cfg_re50
        except ImportError:
            pass
        else:
            cfg_mnet["pretrain"] = False
            cfg_re50["pretrain"] = False
            _retinaface_pretrain_patched = True

    if needs_landmarks and not _star_dirs_patched:
        import openface.STAR.conf.alignment as star_alignment

        star_work_dir = Path(tempfile.gettempdir()) / "scuro_openface_star"
        original_init = star_alignment.Alignment.__init__

        def patched_init(self, args):
            original_init(self, args)
            self.ckpt_dir = str(star_work_dir)
            self.work_dir = os.path.join(
                self.ckpt_dir, self.data_definition, self.folder
            )
            self.model_dir = os.path.join(self.work_dir, "model")
            self.log_dir = os.path.join(self.work_dir, "log")

        star_alignment.Alignment.__init__ = patched_init
        original_init_instance = star_alignment.Alignment.init_instance

        def patched_init_instance(self):
            root = logging.getLogger()
            level, handlers = root.level, list(root.handlers)
            try:
                original_init_instance(self)
            finally:
                for handler in list(root.handlers):
                    if handler not in handlers:
                        root.removeHandler(handler)
                        handler.close()
                root.setLevel(level)

        star_alignment.Alignment.init_instance = patched_init_instance
        _star_dirs_patched = True


@register_representation([ModalityType.IMAGE, ModalityType.VIDEO])
@register_expensive_representation([ModalityType.IMAGE, ModalityType.VIDEO])
class OpenFace(UnimodalRepresentation):
    supports_aggregation_pushdown = True
    cache_in_worker = True

    MODEL_REPOSITORY = "nutPace/openface_weights"
    FACE_MODEL_FILENAME = "Alignment_RetinaFace.pth"
    MULTITASK_MODEL_FILENAME = "MTL_backbone.pth"
    LANDMARK_MODEL_FILENAME = "Landmark_98.pkl"

    FEATURE_SETS = ("landmarks", "behavioral", "multitask", "backbone", "all")
    DEFAULT_FEATURE_SET = "landmarks"
    NUM_LANDMARKS = 98
    BACKBONE_DIM = 1280

    ACTION_UNIT_INTENSITIES = ("01", "06", "17", "25", "26", "02", "12", "15")
    BEHAVIORAL_COLUMNS = (
        "gaze_yaw",
        "gaze_pitch",
        *(f"AU{action_unit}_r" for action_unit in ACTION_UNIT_INTENSITIES),
    )
    EMOTION_COLUMNS = (
        "emotion_neutral",
        "emotion_happy",
        "emotion_sad",
        "emotion_surprise",
        "emotion_fear",
        "emotion_disgust",
        "emotion_anger",
        "emotion_contempt",
    )
    MULTITASK_COLUMNS = BEHAVIORAL_COLUMNS + EMOTION_COLUMNS
    LANDMARK_COLUMNS = tuple(
        coordinate
        for landmark_id in range(NUM_LANDMARKS)
        for coordinate in (f"landmark_{landmark_id}_x", f"landmark_{landmark_id}_y")
    )
    DETECTION_COLUMNS = (
        "face_x1",
        "face_y1",
        "face_x2",
        "face_y2",
        "face_confidence",
        *(
            coordinate
            for landmark_id in range(5)
            for coordinate in (
                f"retinaface_landmark_{landmark_id}_x",
                f"retinaface_landmark_{landmark_id}_y",
            )
        ),
    )
    BACKBONE_COLUMNS = tuple(
        f"backbone_{feature_id}" for feature_id in range(BACKBONE_DIM)
    )

    FEATURE_SET_DIMS = {
        "behavioral": len(BEHAVIORAL_COLUMNS),
        "multitask": len(MULTITASK_COLUMNS),
        "landmarks": len(MULTITASK_COLUMNS) + len(LANDMARK_COLUMNS),
        "backbone": len(BACKBONE_COLUMNS),
        "all": (
            len(MULTITASK_COLUMNS)
            + len(DETECTION_COLUMNS)
            + len(LANDMARK_COLUMNS)
            + len(BACKBONE_COLUMNS)
        ),
    }
    FEATURE_COLUMNS = MULTITASK_COLUMNS + LANDMARK_COLUMNS
    FEATURE_DIM = len(FEATURE_COLUMNS)

    FRAME_BATCH_SIZE = 32
    MAX_SIDE = 1024
    SIZE_WINDOW_CHUNKS = 8
    DETECTOR_MEAN_BGR = (104, 117, 123)

    FACE_MODEL_MEMORY_BYTES = 8 * 1024 * 1024
    MULTITASK_MODEL_MEMORY_BYTES = 128 * 1024 * 1024
    LANDMARK_MODEL_MEMORY_BYTES = 192 * 1024 * 1024
    CPU_RUNTIME_OVERHEAD_BYTES = 128 * 1024 * 1024
    GPU_RUNTIME_OVERHEAD_BYTES = 64 * 1024 * 1024
    LANDMARK_RUNTIME_BYTES = 2400 * 1024 * 1024
    DETECTOR_BYTES_PER_PIXEL = 144
    LANDMARK_BYTES_PER_FRAME = 52 * 1024 * 1024
    MULTITASK_BYTES_PER_FRAME = 26 * 1024 * 1024

    def __init__(
        self,
        feature_set=DEFAULT_FEATURE_SET,
        confidence_threshold=0.02,
        nms_threshold=0.4,
        vis_threshold=0.5,
        max_side=MAX_SIDE,
        params=None,
    ):
        if params is not None:
            feature_set = params.get("feature_set", feature_set)
            confidence_threshold = params.get(
                "confidence_threshold", confidence_threshold
            )
            nms_threshold = params.get("nms_threshold", nms_threshold)
            vis_threshold = params.get("vis_threshold", vis_threshold)
            max_side = params.get("max_side", max_side)

        parameters = {
            "feature_set": list(self.FEATURE_SETS),
            "confidence_threshold": [0.01, 0.02, 0.05],
            "nms_threshold": [0.3, 0.4, 0.5],
            "vis_threshold": [0.4, 0.5, 0.6],
        }
        super().__init__("OpenFace", ModalityType.EMBEDDING, parameters)
        self.feature_set = feature_set
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.vis_threshold = float(vis_threshold)
        self.max_side = int(max_side)
        self.params = params
        self.data_type = np.float32
        self._gpu_id = None
        self.device = get_device()
        self._face_detector = None
        self._multitask_predictor = None
        self._landmark_detector = None
        self._backbone_hook = None
        self._backbone_output = None

    @property
    def feature_set(self):
        return self._feature_set

    @feature_set.setter
    def feature_set(self, feature_set):
        if feature_set not in self.FEATURE_SETS:
            raise ValueError(
                f"Unknown OpenFace feature set '{feature_set}'. "
                f"Expected one of: {', '.join(self.FEATURE_SETS)}"
            )
        self._feature_set = feature_set

    @property
    def feature_dim(self):
        return self.FEATURE_SET_DIMS[self.feature_set]

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        self._gpu_id = gpu_id
        self.device = get_device(gpu_id)
        if self._backbone_hook is not None:
            self._backbone_hook.remove()
        self._face_detector = None
        self._multitask_predictor = None
        self._landmark_detector = None
        self._backbone_hook = None
        self._backbone_output = None

    def get_output_stats(self, input_stats) -> RepresentationStats:
        if self.params and "_pushdown_aggregation" in self.params:
            return RepresentationStats(
                input_stats.num_instances,
                (self.feature_dim,),
                aggregate_dim=None,
                dtype=self.data_type,
            )

        if isinstance(input_stats, VideoStats):
            return RepresentationStats(
                input_stats.num_instances,
                (input_stats.max_length, self.feature_dim),
                dtype=self.data_type,
                container=CONTAINER_LIST,
            )

        if isinstance(input_stats, RepresentationStats):
            return RepresentationStats(
                input_stats.num_instances,
                (input_stats.output_shape[0], self.feature_dim),
                dtype=self.data_type,
                container=CONTAINER_LIST,
            )

        return RepresentationStats(
            input_stats.num_instances,
            (self.feature_dim,),
            aggregate_dim=None,
            dtype=self.data_type,
            container=CONTAINER_LIST,
        )

    def estimate_output_memory_bytes(self, input_stats) -> int:
        stats = self.get_output_stats(input_stats)
        payload = int(
            input_stats.num_instances
            * np.prod(stats.output_shape)
            * np.dtype(self.data_type).itemsize
        )
        return int(
            PY_LIST_HEADER_BYTES
            + input_stats.num_instances * (NP_ARRAY_HEADER_BYTES + PY_LIST_SLOT_BYTES)
            + payload
        )

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        max_height = int(getattr(input_stats, "max_height", 224))
        max_width = int(getattr(input_stats, "max_width", 224))
        max_channels = int(getattr(input_stats, "max_channels", 3))
        float_size = np.dtype(np.float32).itemsize

        if isinstance(input_stats, VideoStats):
            video_bytes = (
                int(input_stats.max_length)
                * max_height
                * max_width
                * max_channels
                * stats_itemsize(input_stats)
            )
            sample_bytes = 2 * video_bytes + 2 * max_height * max_width * 3 * float_size
        else:
            sample_bytes = 2 * max_height * max_width * max_channels

        canvas_pixels = min(max_height, self.max_side) * min(max_width, self.max_side)
        window_bytes = (
            self.FRAME_BATCH_SIZE * self.SIZE_WINDOW_CHUNKS * canvas_pixels * 3
        )
        chunk_bytes = window_bytes + (
            self.FRAME_BATCH_SIZE * canvas_pixels * 3 * 2 * float_size
        )

        model_bytes = self.FACE_MODEL_MEMORY_BYTES + self.MULTITASK_MODEL_MEMORY_BYTES
        batch_bytes = self.MULTITASK_BYTES_PER_FRAME
        runtime_bytes = self.GPU_RUNTIME_OVERHEAD_BYTES
        if self.feature_set in ("landmarks", "all"):
            model_bytes += self.LANDMARK_MODEL_MEMORY_BYTES
            batch_bytes += self.LANDMARK_BYTES_PER_FRAME
            runtime_bytes += self.LANDMARK_RUNTIME_BYTES

        output_bytes = self.estimate_output_memory_bytes(input_stats)
        return {
            "cpu_peak_bytes": int(
                output_bytes
                + sample_bytes
                + chunk_bytes
                + model_bytes
                + self.CPU_RUNTIME_OVERHEAD_BYTES
            ),
            "gpu_peak_bytes": int(
                model_bytes
                + self.FRAME_BATCH_SIZE
                * (canvas_pixels * self.DETECTOR_BYTES_PER_PIXEL + batch_bytes)
                + runtime_bytes
            ),
        }

    def transform(self, modality, aggregation=None):
        if modality.modality_type not in (ModalityType.IMAGE, ModalityType.VIDEO):
            raise ValueError("OpenFace supports only image and video modalities")

        is_image = modality.modality_type == ModalityType.IMAGE
        if is_image:
            embeddings = self._extract_image_features(modality.data)
        else:
            embeddings = []
            lengths = get_sequence_lengths(modality.data, modality.metadata)
            for owner_id, (frames, length) in enumerate(zip(modality.data, lengths)):
                features = self._extract_video_features(frames[:length], owner_id)
                embeddings.append(
                    self._aggregate(features, aggregation)
                    if aggregation is not None
                    else features
                )

            if aggregation is not None:
                embeddings = np.stack(embeddings).astype(np.float32, copy=False)

        if is_image and aggregation is not None:
            embeddings = np.stack(
                [
                    self._aggregate(feature[None, :], aggregation)
                    for feature in embeddings
                ]
            ).astype(np.float32, copy=False)

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        transformed_modality.data_type = np.float32
        transformed_modality.aggregate_dim = (
            None if aggregation is not None or is_image else (0,)
        )
        transformed_modality.data = embeddings
        return transformed_modality

    def _extract_image_features(self, images):
        return list(self._extract_streamed_features(images))

    def _extract_video_features(self, frames, owner_id):
        if len(frames) == 0:
            raise ValueError(f"Video instance {owner_id} contains no frames")
        return self._extract_streamed_features(frames)

    def _extract_features(self, image):
        return self._extract_batch_features([self._prepare(image)])[0]

    def _extract_streamed_features(self, samples):
        features = np.zeros((len(samples), self.feature_dim), dtype=np.float32)
        window = self.FRAME_BATCH_SIZE * self.SIZE_WINDOW_CHUNKS
        for window_start in range(0, len(samples), window):
            indices = range(window_start, min(window_start + window, len(samples)))
            prepared = {index: self._prepare(samples[index]) for index in indices}
            order = sorted(indices, key=lambda index: prepared[index].shape[:2])
            for start, stop in self._chunks(len(order)):
                chunk = order[start:stop]
                features[chunk] = self._extract_batch_features(
                    [prepared[index] for index in chunk]
                )
        return features

    def _prepare(self, image):
        image = self._to_bgr(image)
        height, width = image.shape[:2]
        longer = max(height, width)
        if longer <= self.max_side:
            return image
        scale = self.max_side / longer
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def _pad_to(self, image, height, width):
        pad_bottom = height - image.shape[0]
        pad_right = width - image.shape[1]
        if not pad_bottom and not pad_right:
            return image
        return cv2.copyMakeBorder(
            image,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=self.DETECTOR_MEAN_BGR,
        )

    def _extract_batch_features(self, images_bgr):
        self._load_models()
        with _without_tf32():
            return self._extract_batch_features_fp32(images_bgr)

    def _extract_batch_features_fp32(self, images_bgr):
        features = np.zeros((len(images_bgr), self.feature_dim), dtype=np.float32)
        faces, detections = self._detect_faces(images_bgr)
        with_face = [
            index
            for index, face in enumerate(faces)
            if face is not None and face.size > 0
        ]
        if not with_face:
            return features

        needs_backbone = self.feature_set in ("backbone", "all")
        emotion, gaze, action_units, backbone = self._predict_multitask(
            [faces[index] for index in with_face], needs_backbone
        )
        landmarks = None
        if self.feature_set in ("landmarks", "all"):
            landmarks = self._landmark_features(
                [images_bgr[index] for index in with_face],
                [detections[index] for index in with_face],
            )

        for row, index in enumerate(with_face):
            behavioral = self._behavioral_features(gaze[row], action_units[row])
            if self.feature_set == "behavioral":
                features[index] = behavioral
                continue
            multitask = np.concatenate(
                (
                    behavioral,
                    self._checked_vector(
                        emotion[row], len(self.EMOTION_COLUMNS), "emotion"
                    ),
                )
            )
            if self.feature_set == "multitask":
                features[index] = multitask
                continue
            backbone_row = None
            if needs_backbone:
                backbone_row = self._checked_vector(
                    backbone[row], self.BACKBONE_DIM, "backbone"
                )
                if self.feature_set == "backbone":
                    features[index] = backbone_row
                    continue
            if self.feature_set == "landmarks":
                features[index] = np.concatenate((multitask, landmarks[row]))
                continue
            detection = self._detection_features(
                detections[index], images_bgr[index].shape
            )
            features[index] = np.concatenate(
                (multitask, detection, landmarks[row], backbone_row)
            )
        return features

    def _behavioral_features(self, gaze_output, action_unit_output):
        gaze = self._checked_vector(gaze_output, 2, "gaze")
        action_units = self._checked_vector(
            action_unit_output, len(self.ACTION_UNIT_INTENSITIES), "action-unit"
        )
        return np.concatenate((gaze, action_units)).astype(np.float32, copy=False)

    def _chunks(self, count):
        for start in range(0, count, self.FRAME_BATCH_SIZE):
            yield start, min(start + self.FRAME_BATCH_SIZE, count)

    def _detect_faces(self, images_bgr):
        """Return the top-scoring face crop and its 15-value detection per image.

        Only the highest-scoring box is used downstream, and NMS always keeps
        the highest-scoring box first, so NMS is skipped: an argmax over the
        scores selects the same detection and only that prior is decoded.
        """
        from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
        from openface.Pytorch_Retinaface.utils.box_utils import decode, decode_landm

        detector = self._face_detector
        device = detector.device
        variance = detector.cfg["variance"]
        mean = torch.tensor(self.DETECTOR_MEAN_BGR, dtype=torch.float32, device=device)
        faces = [None] * len(images_bgr)
        detections = [None] * len(images_bgr)
        priors_by_shape = {}

        for start, stop in self._chunks(len(images_bgr)):
            chunk = range(start, stop)
            height = max(images_bgr[i].shape[0] for i in chunk)
            width = max(images_bgr[i].shape[1] for i in chunk)
            if (height, width) not in priors_by_shape:
                priors_by_shape[(height, width)] = (
                    PriorBox(detector.cfg, image_size=(height, width))
                    .forward()
                    .to(device)
                    .data
                )
            priors = priors_by_shape[(height, width)]
            box_scale = torch.tensor(
                [width, height] * 2, dtype=torch.float32, device=device
            )
            landmark_scale = torch.tensor(
                [width, height] * 5, dtype=torch.float32, device=device
            )

            batch = np.stack(
                [self._pad_to(images_bgr[i], height, width) for i in chunk]
            )
            batch = torch.from_numpy(batch).to(device)
            batch = (batch.float() - mean).permute(0, 3, 1, 2).contiguous()
            with torch.no_grad():
                locations, confidences, landmark_offsets = detector.model(batch)

                rows = torch.arange(len(chunk), device=device)
                scores, best = confidences[:, :, 1].max(dim=1)
                best_priors = priors[best]
                boxes = decode(locations[rows, best], best_priors, variance)
                points = decode_landm(
                    landmark_offsets[rows, best], best_priors, variance
                )
                best_dets = torch.cat(
                    (boxes * box_scale, scores[:, None], points * landmark_scale), dim=1
                )
            best_dets = best_dets.cpu().numpy()

            for row, index in enumerate(chunk):
                det = best_dets[row]
                if (
                    det[4] <= detector.confidence_threshold
                    or det[4] < detector.vis_threshold
                ):
                    continue
                bbox = det[:4].astype(int)
                faces[index] = images_bgr[index][bbox[1] : bbox[3], bbox[0] : bbox[2]]
                detections[index] = det
        return faces, detections

    def _predict_multitask(self, faces, needs_backbone):
        predictor = self._multitask_predictor
        if needs_backbone:
            self._ensure_backbone_hook()
        outputs = ([], [], [], [])
        for start, stop in self._chunks(len(faces)):
            batch = torch.stack(
                [
                    predictor.transform(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    for face in faces[start:stop]
                ]
            ).to(predictor.device)
            self._backbone_output = None
            with torch.no_grad():
                emotion, gaze, action_units = predictor.model(batch)
            for output, values in zip(
                outputs, (emotion, gaze, action_units, self._backbone_output)
            ):
                if values is not None:
                    output.append(values.detach().cpu().numpy())
        emotion, gaze, action_units, backbone = (
            np.concatenate(output) if output else None for output in outputs
        )
        return emotion, gaze, action_units, backbone

    def _landmark_features(self, images_bgr, detections):
        alignment = self._landmark_detector.alignment
        features = np.zeros(
            (len(images_bgr), len(self.LANDMARK_COLUMNS)), dtype=np.float32
        )
        max_value = torch.tensor(255.0, device=alignment.device)
        indices, matrices = [], []
        for index, det in enumerate(detections):
            if det[4] < self.vis_threshold:
                continue
            x1, y1, x2, y2 = det[:4].astype(int)
            scale = min(x2 - x1, y2 - y1) / 200 * 1.05
            indices.append(index)
            matrices.append(
                alignment.getCropMatrix.process(
                    float(scale), float((x2 + x1) / 2), float((y2 + y1) / 2)
                )
            )

        for start, stop in self._chunks(len(indices)):
            chunk = indices[start:stop]
            crops = np.stack(
                [
                    alignment.transformPerspective.process(images_bgr[index], matrix)
                    for index, matrix in zip(chunk, matrices[start:stop])
                ]
            )
            crops = torch.from_numpy(crops).to(alignment.device)
            # A device-side divisor keeps CUDA from rewriting the division as a
            # reciprocal multiply, so the inputs match OpenFace's CPU preprocessing.
            crops = crops.float().permute(0, 3, 1, 2) / max_value * 2.0 - 1.0
            with torch.no_grad():
                output = alignment.alignment(crops)
            points = alignment.denorm_points(output[-1]).cpu().numpy()
            if points.shape[1:] != (self.NUM_LANDMARKS, 2):
                raise RuntimeError(
                    "OpenFace 3.0 returned unexpected landmark dimensions: "
                    f"{points.shape[1:]}"
                )
            inverse = np.linalg.inv(np.stack(matrices[start:stop]))
            x, y = points[..., 0], points[..., 1]
            image_points = np.stack(
                (
                    inverse[:, 0, 0, None] * x
                    + inverse[:, 0, 1, None] * y
                    + inverse[:, 0, 2, None],
                    inverse[:, 1, 0, None] * x
                    + inverse[:, 1, 1, None] * y
                    + inverse[:, 1, 2, None],
                ),
                axis=-1,
            )
            image_sizes = np.array(
                [images_bgr[index].shape[1::-1] for index in chunk], dtype=np.float32
            )
            image_points /= image_sizes[:, None, :]
            features[chunk] = image_points.reshape(len(chunk), -1)
        return features

    @classmethod
    def _detection_features(cls, detection, image_shape):
        detection = np.asarray(detection, dtype=np.float32)
        if detection.size < len(cls.DETECTION_COLUMNS):
            raise RuntimeError(
                "OpenFace 3.0 returned unexpected face-detection dimensions: "
                f"{detection.size}"
            )

        detection = detection[: len(cls.DETECTION_COLUMNS)].copy()
        height, width = image_shape[:2]
        detection[[0, 2, 5, 7, 9, 11, 13]] /= width
        detection[[1, 3, 6, 8, 10, 12, 14]] /= height
        return detection

    def _ensure_backbone_hook(self):
        if self._backbone_hook is not None:
            return

        def capture_backbone(_module, _inputs, output):
            self._backbone_output = output

        self._backbone_hook = (
            self._multitask_predictor.model.base_model.register_forward_hook(
                capture_backbone
            )
        )

    def _load_models(self):
        needs_landmarks = self.feature_set in ("landmarks", "all")
        models_ready = (
            self._face_detector is not None
            and self._multitask_predictor is not None
            and (not needs_landmarks or self._landmark_detector is not None)
        )
        if models_ready:
            return

        try:
            from huggingface_hub import snapshot_download
            from openface.face_detection import FaceDetector
            from openface.multitask_model import MultitaskPredictor

            if needs_landmarks:
                from openface.landmark_detection import LandmarkDetector
        except ImportError as error:
            raise ImportError(
                "OpenFace 3.0 is required for this representation. "
                "Install it with 'pip install openface-test'."
            ) from error

        _patch_openface_package_defaults(needs_landmarks)

        weight_files = [self.FACE_MODEL_FILENAME, self.MULTITASK_MODEL_FILENAME]
        if needs_landmarks:
            weight_files.append(self.LANDMARK_MODEL_FILENAME)
        weights_directory = Path(
            snapshot_download(
                repo_id=self.MODEL_REPOSITORY,
                allow_patterns=weight_files,
            )
        )

        class ArrayFaceDetector(FaceDetector):
            def preprocess_image(self, image, resize=1.0):
                image_raw = np.ascontiguousarray(image)
                detector_input = np.float32(image_raw)
                if resize != 1:
                    detector_input = cv2.resize(
                        detector_input,
                        None,
                        fx=resize,
                        fy=resize,
                        interpolation=cv2.INTER_LINEAR,
                    )
                detector_input -= (104, 117, 123)
                detector_input = detector_input.transpose(2, 0, 1)
                detector_input = (
                    torch.from_numpy(detector_input).unsqueeze(0).to(self.device)
                )
                return detector_input, image_raw

        device = str(self.device)
        if self._face_detector is None:
            self._face_detector = ArrayFaceDetector(
                model_path=str(weights_directory / self.FACE_MODEL_FILENAME),
                device=device,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                vis_threshold=self.vis_threshold,
            )
        if self._multitask_predictor is None:
            self._multitask_predictor = MultitaskPredictor(
                model_path=str(weights_directory / self.MULTITASK_MODEL_FILENAME),
                device=device,
            )
        if needs_landmarks and self._landmark_detector is None:
            landmark_device = self.device.type
            device_ids = (
                [-1]
                if landmark_device == "cpu"
                else [self.device.index if self.device.index is not None else 0]
            )
            self._landmark_detector = LandmarkDetector(
                model_path=str(weights_directory / self.LANDMARK_MODEL_FILENAME),
                device=landmark_device,
                device_ids=device_ids,
            )

    @staticmethod
    def _checked_vector(output, expected_size, output_name):
        if output is None:
            size = 0
        else:
            if hasattr(output, "detach"):
                output = output.detach().cpu().numpy()
            output = np.asarray(output, dtype=np.float32).reshape(-1)
            size = output.size
        if size != expected_size:
            raise RuntimeError(
                f"OpenFace 3.0 returned unexpected {output_name} dimensions: {size}"
            )
        return output

    @staticmethod
    def _aggregate(features, aggregation):
        return np.asarray(aggregation.execute(features), dtype=np.float32)

    @staticmethod
    def _to_bgr(image):
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim not in (2, 3):
            raise ValueError(
                f"Expected an image tensor with 2 or 3 dimensions, got {image.shape}"
            )

        if image.dtype in (np.float16, np.float32) and image.size:
            values = image.astype(np.float32, copy=False)
            low, high = values.min(), values.max()
            if np.isfinite(low) and np.isfinite(high):
                if low >= 0.0 and high <= 1.0:
                    values = values * np.float32(255.0)
                    if image.dtype == np.float16:
                        values = values.astype(np.float16).astype(np.float32)
                image = np.clip(values, 0, 255).astype(np.uint8)
        if np.issubdtype(image.dtype, np.floating):
            finite = image[np.isfinite(image)]
            if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
                image = image * 255.0
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        image = np.clip(image, 0, 255).astype(np.uint8, copy=False)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError(
                f"Expected 1, 3, or 4 image channels, got {image.shape[2]}"
            )
        return np.ascontiguousarray(image)

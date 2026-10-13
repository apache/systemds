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
import os
import pickle
from bisect import bisect_right
from collections.abc import Sequence
from contextlib import contextmanager

import numpy as np
import torch


def pool_transformer_output(hidden_state, attention_mask, use_cls=False):
    """Pool transformer tokens without including padding tokens."""
    if hidden_state.ndim == 2:
        return hidden_state
    if hidden_state.ndim != 3:
        raise ValueError(
            f"Unexpected transformer output shape: {tuple(hidden_state.shape)}"
        )
    if use_cls:
        return hidden_state[:, 0, :]

    mask = attention_mask.unsqueeze(-1).to(
        device=hidden_state.device, dtype=hidden_state.dtype
    )
    token_count = mask.sum(dim=1).clamp_min(1)
    return (hidden_state * mask).sum(dim=1) / token_count


def aggregate_chunk_embeddings(embeddings, aggregation):
    """Aggregate all chunks of one instance while they are still on the GPU."""
    name = aggregation.aggregation_function
    if name == "mean":
        return embeddings.mean(dim=0)
    if name == "max":
        return embeddings.max(dim=0).values
    if name == "min":
        return embeddings.min(dim=0).values
    if name == "sum":
        return embeddings.sum(dim=0)
    if name == "median":
        return torch.quantile(embeddings, 0.5, dim=0)
    if name == "mode":
        return embeddings.mode(dim=0).values
    raise ValueError(f"Unsupported aggregation function: {name}")


class LengthBucketBatchSampler(torch.utils.data.Sampler):

    def __init__(self, lengths, batch_size, exact=False, max_batch_items=None):
        self.lengths = [int(length) for length in lengths]
        self.batch_size = max(1, int(batch_size))
        self.exact = exact
        self.max_batch_items = (
            None if max_batch_items is None else max(1, int(max_batch_items))
        )
        self._batches = self._build_batches()

    def _batch_size_for(self, length):
        if self.max_batch_items is None:
            return self.batch_size
        return max(1, min(self.batch_size, self.max_batch_items // max(length, 1)))

    def _build_batches(self):
        indices = sorted(range(len(self.lengths)), key=lambda i: (self.lengths[i], i))
        if not self.exact:
            batches = []
            batch = []
            for index in indices:
                if batch and len(batch) >= self._batch_size_for(self.lengths[index]):
                    batches.append(batch)
                    batch = []
                batch.append(index)
            if batch:
                batches.append(batch)
            return batches

        batches = []
        start = 0
        while start < len(indices):
            length = self.lengths[indices[start]]
            end = start
            while end < len(indices) and self.lengths[indices[end]] == length:
                end += 1
            batch_size = self._batch_size_for(length)
            batches.extend(
                indices[offset : min(offset + batch_size, end)]
                for offset in range(start, end, batch_size)
            )
            start = end
        return batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class OwnedSequenceDataset(torch.utils.data.Dataset):
    """Sequence samples with stable chunk and owner identifiers."""

    def __init__(self, samples, owner_ids=None):
        self.samples = list(samples)
        if owner_ids is None:
            owner_ids = range(len(self.samples))
        self.owner_ids = [int(owner_id) for owner_id in owner_ids]
        if len(self.samples) != len(self.owner_ids):
            raise ValueError("Each sequence must have exactly one owner_id")

    def __getitem__(self, chunk_id):
        return self.samples[chunk_id], self.owner_ids[chunk_id], chunk_id

    def __len__(self):
        return len(self.samples)


class FlattenedSequence(Sequence):
    """Lazy flattened view over per-owner sequences."""

    def __init__(self, sequences, lengths):
        self.sequences = sequences
        self.lengths = tuple(int(length) for length in lengths)
        if len(self.sequences) != len(self.lengths):
            raise ValueError("Each sequence must have exactly one length")

        self.offsets = [0]
        for length in self.lengths:
            self.offsets.append(self.offsets[-1] + length)
        self._cached_owner = None
        self._cached_sequence = None

    @property
    def owner_ids(self):
        return [
            owner_id
            for owner_id, length in enumerate(self.lengths)
            for _ in range(length)
        ]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        owner_id = bisect_right(self.offsets, index) - 1
        if owner_id != self._cached_owner:
            self._cached_sequence = self.sequences[owner_id]
            self._cached_owner = owner_id
        return self._cached_sequence[index - self.offsets[owner_id]]

    def __len__(self):
        return self.offsets[-1]


class LazyModelMixin:
    """Loads the pretrained model on first use of ``self.model``.

    The optimizer and scheduler instantiate operators in the main process only
    to read their parameter grids, output shapes and memory estimates, so
    those must come from class-level constants rather than the model.
    """

    _model = None

    def _build_model(self):
        raise NotImplementedError

    @property
    def model(self):
        if self._model is None:
            self._model = self._build_model()
        return self._model

    @model.setter
    def model(self, model):
        self._model = model


@contextmanager
def _parameters_on_meta():
    """Move every parameter to the meta device as soon as it is registered.

    Weight initializers then run on meta tensors and cost nothing. Buffers stay
    real because some modules compute them in ``__init__`` (e.g. ImageBind's
    causal mask) and a non-persistent buffer is never restored from a
    checkpoint. Whole-module ``torch.device("meta")`` would also break modules
    that call ``.item()`` on tensors during construction.
    """
    register_parameter = torch.nn.Module.register_parameter

    def register_on_meta(module, name, param):
        register_parameter(module, name, param)
        if param is not None and not param.is_meta:
            module._parameters[name] = torch.nn.Parameter(
                param.to("meta"), requires_grad=param.requires_grad
            )

    torch.nn.Module.register_parameter = register_on_meta
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = register_parameter


def build_pretrained(build_fn, state_dict):
    """Build ``build_fn()`` without random weight init and load ``state_dict``.

    Initializing a large model only to overwrite every weight from the
    checkpoint dominates load time (about 12 s of ImageBind's 20 s).
    """
    with _parameters_on_meta():
        model = build_fn()
    model.load_state_dict(state_dict, strict=True, assign=True)
    missing = [
        name
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]
        if tensor.is_meta
    ]
    if missing:
        raise RuntimeError(f"Checkpoint did not provide weights for {missing}")
    return model


def load_checkpoint(path):
    """Load a state dict on the CPU, memory-mapped when the format allows it."""
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except RuntimeError as error:
        # Checkpoints in the legacy (pre zip) format cannot be memory-mapped.
        if "mmap" not in str(error):
            raise
        return torch.load(path, map_location="cpu", weights_only=True)


def torchvision_state_dict(weights):
    """Load torchvision ``weights`` from the torch hub cache."""
    from urllib.parse import urlparse

    file_name = os.path.basename(urlparse(weights.url).path)
    path = os.path.join(torch.hub.get_dir(), "checkpoints", file_name)
    if not os.path.exists(path):
        return weights.get_state_dict(progress=False, check_hash=True)
    return load_checkpoint(path)


def build_torchvision_model(factory, weights, **kwargs):
    """Equivalent to ``factory(weights=weights, **kwargs)`` without random init."""
    kwargs.setdefault("num_classes", len(weights.meta["categories"]))
    return build_pretrained(
        lambda: factory(weights=None, **kwargs), torchvision_state_dict(weights)
    )


def get_sequence_lengths(sequences, metadata):
    if len(metadata) == len(sequences) and all("length" in md for md in metadata):
        return [int(md["length"]) for md in metadata]
    return [len(sequence) for sequence in sequences]


def flatten_owned_sequences(sequences, lengths=None):
    """Flatten per-owner sequences while retaining their owner identifiers."""
    if lengths is not None:
        samples = FlattenedSequence(sequences, lengths)
        return samples, samples.owner_ids

    samples = []
    owner_ids = []
    for owner_id, owner_samples in enumerate(sequences):
        samples.extend(owner_samples)
        owner_ids.extend([owner_id] * len(owner_samples))
    return samples, owner_ids


def pin_memory_for(device):
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.is_available()


def move_batch_to_device(batch, device):
    non_blocking = pin_memory_for(device)
    return {
        key: (
            value.to(device, non_blocking=non_blocking)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


@contextmanager
def inference_context(device):
    """Enable inference-only execution and mixed precision on CUDA."""
    device = torch.device(device)
    with torch.inference_mode():
        if device.type != "cuda":
            yield
            return
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield


transformer_inference_context = inference_context


class OwnerAccumulator:
    """Collect chunk vectors on-device and restore or aggregate their owners."""

    def __init__(self, num_owners, num_chunks, aggregation=None):
        self.num_owners = int(num_owners)
        self.num_chunks = int(num_chunks)
        self.aggregation = aggregation
        self._values = None
        self._counts = None
        self._owner_ids = None

    @property
    def aggregation_name(self):
        if self.aggregation is None:
            return None
        return self.aggregation.aggregation_function

    def _initialize(self, vectors):
        hidden_dim = vectors.shape[1]
        name = self.aggregation_name
        if name is None or name in ("median", "mode"):
            self._values = torch.empty(
                (self.num_chunks, hidden_dim),
                device=vectors.device,
                dtype=torch.float32,
            )
            self._owner_ids = torch.empty(
                self.num_chunks, device=vectors.device, dtype=torch.long
            )
        else:
            fill = 0.0
            if name == "max":
                fill = -torch.inf
            elif name == "min":
                fill = torch.inf
            self._values = torch.full(
                (self.num_owners, hidden_dim),
                fill,
                device=vectors.device,
                dtype=torch.float32,
            )
            self._counts = torch.zeros(
                self.num_owners, device=vectors.device, dtype=torch.long
            )

    def update(self, vectors, owner_ids, chunk_ids):
        vectors = vectors.detach().float()
        owner_ids = owner_ids.to(device=vectors.device, dtype=torch.long)
        chunk_ids = chunk_ids.to(device=vectors.device, dtype=torch.long)
        if self._values is None:
            self._initialize(vectors)

        name = self.aggregation_name
        if name is None or name in ("median", "mode"):
            self._values.index_copy_(0, chunk_ids, vectors)
            self._owner_ids.index_copy_(0, chunk_ids, owner_ids)
            return

        self._counts.index_add_(
            0, owner_ids, torch.ones_like(owner_ids, dtype=torch.long)
        )
        if name in ("mean", "sum"):
            self._values.index_add_(0, owner_ids, vectors)
        elif name in ("max", "min"):
            indices = owner_ids[:, None].expand_as(vectors)
            self._values.scatter_reduce_(
                0, indices, vectors, reduce=f"a{name}", include_self=True
            )
        else:
            raise ValueError(f"Unsupported aggregation function: {name}")

    def finalize(self, grouped=False):
        if self._values is None:
            return np.empty((self.num_owners, 0), dtype=np.float32)

        name = self.aggregation_name
        if name is not None:
            if name == "mean":
                values = self._values / self._counts.clamp_min(1).unsqueeze(1)
            elif name in ("sum", "max", "min"):
                values = self._values
            else:
                values = torch.stack(
                    [
                        aggregate_chunk_embeddings(
                            self._values[self._owner_ids == owner], self.aggregation
                        )
                        for owner in range(self.num_owners)
                    ]
                )
            return values.cpu().numpy().astype(np.float32, copy=False)

        values = self._values.cpu().numpy().astype(np.float32, copy=False)
        if not grouped:
            return values
        owner_ids = self._owner_ids.cpu().numpy()
        return [values[owner_ids == owner] for owner in range(self.num_owners)]


def dense_instance_batch(data):
    if isinstance(data, np.ndarray):
        if data.ndim == 2 and np.issubdtype(data.dtype, np.number):
            return data
        return None

    if not isinstance(data, (list, tuple)) or len(data) == 0:
        return None

    first = data[0]
    if (
        not isinstance(first, np.ndarray)
        or first.ndim != 1
        or not np.issubdtype(first.dtype, np.number)
    ):
        return None
    for instance in data:
        if not isinstance(instance, np.ndarray) or instance.shape != first.shape:
            return None

    return np.asarray(data)


def pad_sequences(sequences, maxlen=None, dtype="float32", value=0):
    if maxlen is None:
        maxlen = max([len(seq) for seq in sequences])

    result = np.full((len(sequences), maxlen), value, dtype=dtype)

    for i, seq in enumerate(sequences):
        data = seq[:maxlen]
        result[i, : len(data)] = data

    return result


def get_segments(data, key_prefix):
    segments = {}
    counter = 1
    for line in data:
        line = line.replace("\n", "")
        segments[key_prefix + str(counter)] = line
        counter += 1

    return segments


def read_data_from_file(filepath, indices):
    data = {}

    is_dir = True if os.path.isdir(filepath) else False

    if is_dir:
        files = os.listdir(filepath)

        # get file extension
        _, ext = os.path.splitext(files[0])
        for key in indices:
            with open(filepath + key + ext) as segm:
                data.update(get_segments(segm, key + "_"))
    else:
        with open(filepath) as file:
            data.update(get_segments(file, ""))

    return data


def save_embeddings(data, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(data, file)

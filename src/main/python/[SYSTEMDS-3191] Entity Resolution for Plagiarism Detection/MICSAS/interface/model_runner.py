'''
MIT License

Copyright (c) 2021 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Dict, Generic, TypeVar, Union
import torch

from ..models.gnn_model import GNNModel


T = TypeVar('T')


class ModelRunner(ABC, Generic[T]):
    @abstractmethod
    def compute_code_vector(self, preprocessed_cass: T) -> np.ndarray:
        """
        Compute a code vector for preprocessed CASS.
        Return a 1D array with shape (vector_size,).
        """
        pass

    @abstractmethod
    def compute_code_vector_batched(self, preprocessed_cass_batch: Iterable[T]) -> np.ndarray:
        """
        Compute code vectors for a batch of preprocessed CASS.
        Return a 2D array with shape (batch_size, vector_size).
        """
        pass


class GNNRunner(ModelRunner[Tuple[np.ndarray, np.ndarray]]):
    def __init__(self, vocab: Union[str, Dict[str, int]], model_path: str, output_size: int = 128,
                 node_emb_size: int = 128, num_layers: int = 3, device: str = 'cpu') -> None:
        if isinstance(vocab, str):
            with open(vocab, 'rb') as f:
                self.vocab: Dict[str, int] = pickle.load(f)
        else:
            self.vocab = vocab
        self.device = torch.device(device)
        self.model = GNNModel(node_emb_size, len(self.vocab),
                              output_size, num_layers)
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))

    def compute_code_vector(self, preprocessed_cass: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return np.squeeze(self.compute_code_vector_batched([preprocessed_cass]), axis=0)

    def compute_code_vector_batched(self, preprocessed_cass_batch: Iterable[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        num_prev_nodes = 0
        nodes_batch = []
        edges_batch = []
        indices_batch = []
        for i, (nodes, edges) in enumerate(preprocessed_cass_batch):
            num_nodes = nodes.shape[0]
            nodes_batch.append(torch.from_numpy(nodes))
            edges_batch.append(torch.from_numpy(edges) + num_prev_nodes)
            num_prev_nodes += num_nodes
            indices_batch.append(torch.full((num_nodes,), i, dtype=torch.long))

        batched_nodes = torch.cat(nodes_batch).to(self.device)
        batched_edges = torch.cat(edges_batch, dim=1).to(self.device)
        batched_indices = torch.cat(indices_batch).to(self.device)

        with torch.no_grad():
            v: torch.Tensor = self.model(batched_nodes, batched_edges, batched_indices)

        return v.detach().cpu().numpy()

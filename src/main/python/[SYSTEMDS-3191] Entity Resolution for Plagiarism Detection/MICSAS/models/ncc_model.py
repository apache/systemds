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
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class NCCModel(nn.Module):
    def __init__(self, inst2vec_emb, rnn_size, dense_size, output_size, use_i2v_emb):
        super().__init__()
        if use_i2v_emb:
            self.emb = nn.Embedding.from_pretrained(inst2vec_emb, freeze=True)
        else:
            self.emb = nn.Embedding(inst2vec_emb.size(0), inst2vec_emb.size(1))
        self.rnn = nn.LSTM(self.emb.embedding_dim, rnn_size, num_layers=2)
        self.batch_norm = nn.BatchNorm1d(rnn_size)
        self.out = nn.Sequential(
            nn.Linear(rnn_size, dense_size),
            nn.ReLU(),
            nn.Linear(dense_size, output_size)
        )

    def forward(self, seqs):
        seqs = PackedSequence(
            self.emb(seqs.data),
            seqs.batch_sizes,
            seqs.sorted_indices,
            seqs.unsorted_indices
        )

        _, (hn, _) = self.rnn(seqs)
        x = hn[-1]

        x = self.batch_norm(x)

        return self.out(x)

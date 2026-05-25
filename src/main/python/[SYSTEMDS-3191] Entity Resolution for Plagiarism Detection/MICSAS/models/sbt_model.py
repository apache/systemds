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
from torch_scatter import scatter_max, scatter_mean


class SBTModel(nn.Module):
    def __init__(self, token_emb_size, token_vocab_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_emb = nn.Sequential(
            nn.Embedding(token_vocab_size, token_emb_size),
            nn.Dropout(0.5)
        )
        self.rnn = nn.GRU(token_emb_size, hidden_size, bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 4, output_size)
        )

    def forward(self, seqs, indices):
        seqs = PackedSequence(
            self.token_emb(seqs.data),
            seqs.batch_sizes,
            seqs.sorted_indices,
            seqs.unsorted_indices
        )

        _, hn = self.rnn(seqs)

        h = torch.cat((hn[0], hn[1]), dim=1)

        v = torch.cat(
            (
                scatter_mean(h, indices, dim=0),
                scatter_max(h, indices, dim=0)[0]
            ),
            dim=1
        )

        v = self.proj(v)

        return v

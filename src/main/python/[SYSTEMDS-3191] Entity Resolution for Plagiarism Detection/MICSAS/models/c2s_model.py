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
from torch_scatter import scatter_add
from torch_scatter.composite import scatter_softmax


class C2SModel(nn.Module):
    def __init__(self, emb_size, subtoken_vocab_size, node_vocab_size, rnn_size, decoder_size, output_size):
        super().__init__()
        self.subtoken_emb = nn.Embedding(subtoken_vocab_size, emb_size)
        self.node_emb = nn.Embedding(node_vocab_size, emb_size)
        self.path_rnn = nn.LSTM(emb_size, rnn_size // 2, bidirectional=True, batch_first=True)

        self.emb_dropout = nn.Dropout(0.25)

        self.fc = nn.Sequential(
            nn.Linear(emb_size * 2 + rnn_size,
                      decoder_size, bias=False),
            nn.Tanh()
        )

        self.a = nn.Parameter(torch.empty(decoder_size, dtype=torch.float))
        nn.init.uniform_(self.a)

        self.out = nn.Linear(decoder_size, output_size)

    def forward(self, ll_subtokens, ll_indices, rl_subtokens, rl_indices, paths, indices):
        ll_emb = scatter_add(self.subtoken_emb(ll_subtokens), ll_indices, dim=0)
        rl_emb = scatter_add(self.subtoken_emb(rl_subtokens), rl_indices, dim=0)

        _, (h, _) = self.path_rnn(
            PackedSequence(
                self.node_emb(paths.data),
                paths.batch_sizes,
                paths.sorted_indices,
                paths.unsorted_indices
            )
        )  # (2, batch_context_num, rnn_size // 2)
        path_nodes_aggregation = torch.cat((h[0], h[1]), dim=1)  # (batch_context_num, rnn_size)

        context_emb = torch.cat((ll_emb, path_nodes_aggregation, rl_emb), dim=1)
        context_emb = self.emb_dropout(context_emb)

        context_emb = self.fc(context_emb)  # (batch_context_num, decoder_size)

        attn_score = torch.matmul(context_emb, self.a)
        attn_weight = scatter_softmax(attn_score, indices, dim=0)
        weighted_context = context_emb * attn_weight.unsqueeze(1)
        v = scatter_add(weighted_context, indices, dim=0)

        return self.out(v)

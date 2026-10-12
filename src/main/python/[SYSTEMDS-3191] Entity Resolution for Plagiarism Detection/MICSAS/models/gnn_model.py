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
from torch_scatter import scatter_max, scatter_mean


class GNNModel(nn.Module):
    def __init__(self, node_emb_size, node_vocab_size, output_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.node_emb = nn.Sequential(
            nn.Embedding(node_vocab_size, node_emb_size),
            nn.Dropout(0.5)
        )
        self.gnn_layers = nn.ModuleList(
            [RGCNLayer(node_emb_size) for _ in range(num_layers)])
        self.out = nn.Sequential(
            nn.Linear(node_emb_size * 2, output_size)
        )

    def forward(self, nodes, edges, indices):
        h = self.node_emb(nodes)
        for i in range(self.num_layers):
            h = self.gnn_layers[i](h, edges)

        v = torch.cat(
            (
                scatter_mean(h, indices, dim=0),
                scatter_max(h, indices, dim=0)[0]
            ),
            dim=1
        )

        return self.out(v)


class RGCNLayer(nn.Module):
    def __init__(self, node_emb_size):
        super().__init__()
        self.W0 = nn.Linear(node_emb_size, node_emb_size, bias=False)
        self.W1 = nn.Linear(node_emb_size, node_emb_size, bias=False)
        self.W2 = nn.Linear(node_emb_size, node_emb_size, bias=False)

    def forward(self, nodes, edges):
        nbr_msg = torch.cat(
            (self.W1(nodes[edges[0]]), self.W2(nodes[edges[1]])), dim=0)
        msg = scatter_mean(nbr_msg, torch.cat(
            (edges[1], edges[0])), dim=0, dim_size=nodes.size(0))
        return torch.relu(self.W0(nodes) + msg)

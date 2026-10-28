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
import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from systemds.scuro.drsearch.representation_dag import (
    CSEAwareDAGBuilder,
    pushdown_aggregation,
)
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.clip import CLIPText
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.utils import pool_transformer_output


class _BatchEncoding(dict):
    @property
    def data(self):
        return self

    def to(self, device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self


class _PadMixin:
    def pad(self, features, **kwargs):
        rows = [
            {
                key: torch.as_tensor(feature[key])
                for key in ("input_ids", "attention_mask")
            }
            for feature in features
        ]
        padded_length = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.zeros((len(rows), padded_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row["input_ids"])] = row["input_ids"]
            attention_mask[index, : len(row["attention_mask"])] = row["attention_mask"]
        return _BatchEncoding(input_ids=input_ids, attention_mask=attention_mask)


class _Tokenizer(_PadMixin):
    def __call__(self, batch, **kwargs):
        ids = torch.tensor([int(text) for text in batch])
        input_ids = ids.unsqueeze(1).repeat(1, 3)
        attention_mask = torch.tensor([[1, 1, 0]]).repeat(len(batch), 1)
        return _BatchEncoding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            offset_mapping=torch.zeros((len(batch), 3, 2), dtype=torch.long),
        )


class _BertModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        ids = input_ids[:, 0].float()
        hidden = torch.stack(
            (
                torch.stack((ids, ids + 10), dim=1),
                torch.stack((ids + 100, ids + 200), dim=1),
                torch.full((len(ids), 2), 1000.0, device=ids.device),
            ),
            dim=1,
        )
        return SimpleNamespace(last_hidden_state=hidden)


class _DynamicTokenizer(_PadMixin):
    def __call__(self, batch, **kwargs):
        tokens = [[int(token) for token in text.split()] for text in batch]
        max_length = kwargs.get("max_length")
        if max_length is not None:
            tokens = [values[:max_length] for values in tokens]

        if kwargs.get("return_tensors") != "pt":
            return _BatchEncoding(
                input_ids=tokens,
                attention_mask=[[1] * len(values) for values in tokens],
            )

        padded_length = max(len(values) for values in tokens)
        input_ids = torch.zeros((len(tokens), padded_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, values in enumerate(tokens):
            input_ids[row, : len(values)] = torch.tensor(values)
            attention_mask[row, : len(values)] = 1
        return _BatchEncoding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            offset_mapping=torch.zeros(
                (len(tokens), padded_length, 2), dtype=torch.long
            ),
        )


class _IntermediateBertModel(torch.nn.Module):
    def __init__(self, representation):
        super().__init__()
        self.intermediate = torch.nn.Identity()

    def forward(self, input_ids, attention_mask):
        values = input_ids.float()
        hidden = torch.stack((values, values + 10), dim=-1)
        hidden = torch.where(
            attention_mask.unsqueeze(-1).bool(),
            hidden,
            torch.full_like(hidden, 1000),
        )
        return SimpleNamespace(last_hidden_state=self.intermediate(hidden))


class _CLIPProcessor:
    def __call__(self, text, **kwargs):
        ids = torch.tensor([int(value) for value in text])
        return _BatchEncoding(
            input_ids=ids.unsqueeze(1),
            attention_mask=torch.tensor([[1, 1, 0]]).repeat(len(text), 1),
        )


class _CLIPTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate = torch.nn.Identity()

    def forward(self, input_ids, attention_mask):
        ids = input_ids[:, 0].float()
        return self.intermediate(
            torch.stack(
                (
                    torch.stack((ids, ids + 2), dim=1),
                    torch.stack((ids + 2, ids + 4), dim=1),
                    torch.full((len(ids), 2), 1000.0, device=ids.device),
                ),
                dim=1,
            )
        )


class _CLIPModel:
    def __init__(self, representation):
        self.text_model = _CLIPTextModel()


class TestTransformerTextAggregation(unittest.TestCase):
    def test_token_pooling_selects_cls_or_masked_mean(self):
        hidden = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [1000.0, 1000.0]],
                [[5.0, 6.0], [1000.0, 1000.0], [1000.0, 1000.0]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        np.testing.assert_allclose(
            pool_transformer_output(hidden, mask, use_cls=True).numpy(),
            [[1.0, 2.0], [5.0, 6.0]],
        )
        np.testing.assert_allclose(
            pool_transformer_output(hidden, mask).numpy(),
            [[2.0, 3.0], [5.0, 6.0]],
        )

    def test_output_stats_describe_pooled_chunk_vectors(self):
        raw_stats = SimpleNamespace(num_instances=3)
        context_stats = RepresentationStats(3, (5, 77))

        bert_plain = Bert().get_output_stats(raw_stats)
        bert_context = Bert().get_output_stats(context_stats)
        clip_plain = CLIPText().get_output_stats(raw_stats)
        clip_context = CLIPText().get_output_stats(context_stats)

        self.assertEqual(bert_plain.output_shape, (768,))
        self.assertEqual(bert_context.output_shape, (5, 768))
        self.assertEqual(bert_context.aggregate_dim, (0,))
        self.assertEqual(clip_plain.output_shape, (512,))
        self.assertEqual(clip_context.output_shape, (5, 512))
        self.assertEqual(clip_context.aggregate_dim, (0,))

    def test_bert_cls_aggregation_is_independent_of_batch_boundaries(self):
        representation = Bert(batch_size=2, max_seq_length=3)
        result = representation.create_embeddings(
            ["0", "1", "2", "3", "4"],
            _BertModel(),
            _Tokenizer(),
            AggregatedRepresentation("mean"),
        )

        np.testing.assert_allclose(result, [2.0, 12.0])
        self.assertEqual(result.shape, (2,))

    def test_global_batching_returns_one_vector_per_patient(self):
        chunks = ["1", "3 5", "7 9 11", "2", "4 6"]
        owner_ids = [0, 0, 1, 2, 2]
        representation = Bert(layer="intermediate", batch_size=3, max_seq_length=8)
        result = representation.create_embeddings(
            chunks,
            _IntermediateBertModel(representation),
            _DynamicTokenizer(),
            AggregatedRepresentation("mean"),
            owner_ids=owner_ids,
            num_owners=3,
        )

        self.assertEqual(result.shape, (3, 2))

    def test_global_batching_is_invariant_to_batch_size(self):
        chunks = ["1", "3 5", "7 9 11", "2", "4 6"]
        owner_ids = [0, 0, 1, 2, 2]
        results = []
        for batch_size in (1, 2, 4):
            representation = Bert(
                layer="intermediate",
                batch_size=batch_size,
                max_seq_length=8,
            )
            results.append(
                representation.create_embeddings(
                    chunks,
                    _IntermediateBertModel(representation),
                    _DynamicTokenizer(),
                    AggregatedRepresentation("mean"),
                    owner_ids=owner_ids,
                    num_owners=3,
                )
            )

        for result in results[1:]:
            np.testing.assert_allclose(result, results[0])

    def test_dynamic_padding_does_not_change_embeddings(self):
        short = "2 4"
        representation = Bert(layer="intermediate", batch_size=2, max_seq_length=8)
        model = _IntermediateBertModel(representation)
        tokenizer = _DynamicTokenizer()

        alone = representation.create_embeddings([short], model, tokenizer)[0]
        mixed = representation.create_embeddings(
            [short, "10 20 30 40 50"], model, tokenizer
        )[0]

        np.testing.assert_allclose(mixed, alone)

    def test_global_batching_matches_patient_by_patient_batching(self):
        patient_chunks = [["1", "3 5"], ["7 9 11"], ["2", "4 6"]]
        chunks = [chunk for patient in patient_chunks for chunk in patient]
        owner_ids = [
            owner_id for owner_id, patient in enumerate(patient_chunks) for _ in patient
        ]
        aggregation = AggregatedRepresentation("mean")
        representation = Bert(layer="intermediate", batch_size=3, max_seq_length=8)
        model = _IntermediateBertModel(representation)
        tokenizer = _DynamicTokenizer()

        global_result = representation.create_embeddings(
            chunks,
            model,
            tokenizer,
            aggregation,
            owner_ids=owner_ids,
            num_owners=len(patient_chunks),
        )
        per_patient_result = np.stack(
            [
                representation.create_embeddings(patient, model, tokenizer, aggregation)
                for patient in patient_chunks
            ]
        )

        np.testing.assert_allclose(global_result, per_patient_result)

    def test_clip_intermediate_layer_uses_masked_mean_before_chunk_aggregation(self):
        representation = CLIPText(batch_size=2, layer_name="intermediate")
        representation.processor = _CLIPProcessor()
        result = representation.create_text_embeddings(
            ["0", "2", "4"],
            _CLIPModel(representation),
            AggregatedRepresentation("mean"),
        )

        np.testing.assert_allclose(result, [3.0, 5.0])
        self.assertEqual(result.shape, (2,))

    def test_pushdown_skipped_when_plain_output_is_also_needed(self):
        builder = CSEAwareDAGBuilder()
        leaf_id = builder.create_leaf_node("transformer_pushdown")
        bert = Bert()
        bert_id = builder.create_operation_node(
            Bert, [leaf_id], bert.get_current_parameters()
        )
        aggregation = AggregatedRepresentation(
            "mean", target_dimensions=1, aggregate_leading=True
        )
        aggregation_id = builder.create_operation_node(
            AggregatedRepresentation,
            [bert_id],
            aggregation.get_current_parameters(),
        )

        plain_dag = builder.build(bert_id)
        aggregated_dag = builder.build(aggregation_id)
        aggregation_params = copy.deepcopy(
            aggregated_dag.get_node_by_id(aggregation_id).parameters
        )

        pushdown_aggregation([plain_dag, aggregated_dag])

        # Bert runs once; its plain output feeds both DAGs and the aggregation
        # is applied to it instead of being fused into a second Bert run.
        plain_node = plain_dag.get_node_by_id(bert_id)
        self.assertIs(plain_node.operation, Bert)
        self.assertNotIn("_pushdown_aggregation", plain_node.parameters)

        aggregation_node = aggregated_dag.get_node_by_id(aggregation_id)
        self.assertIs(aggregation_node.operation, AggregatedRepresentation)
        self.assertEqual(aggregation_node.inputs, [bert_id])
        self.assertEqual(aggregation_node.parameters, aggregation_params)
        self.assertIs(aggregated_dag.get_node_by_id(bert_id), plain_node)

    def test_clip_text_supports_aggregation_pushdown(self):
        builder = CSEAwareDAGBuilder()
        leaf_id = builder.create_leaf_node("clip_pushdown")
        clip_id = builder.create_operation_node(
            CLIPText, [leaf_id], CLIPText().get_current_parameters()
        )
        aggregation = AggregatedRepresentation("max", target_dimensions=1)
        aggregation_id = builder.create_operation_node(
            AggregatedRepresentation,
            [clip_id],
            aggregation.get_current_parameters(),
        )
        dag = builder.build(aggregation_id)

        pushdown_aggregation([dag])

        pushed_node = dag.get_node_by_id(aggregation_id)
        self.assertIs(pushed_node.operation, CLIPText)
        pushed_aggregation = AggregatedRepresentation(
            params=pushed_node.parameters["_pushdown_aggregation"]
        )
        self.assertEqual(pushed_aggregation.aggregation_function, "max")
        self.assertIsNone(dag.get_node_by_id(clip_id))


if __name__ == "__main__":
    unittest.main()

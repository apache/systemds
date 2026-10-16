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

import json
import unittest

from systemds.scuro.drsearch.algebra import ALL_LAWS, canonical_id
from systemds.scuro.drsearch.representation_dag import RepresentationDAGBuilder
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.hadamard import Hadamard
from systemds.scuro.representations.lstm import LSTM
from systemds.scuro.representations.max import RowMax
from systemds.scuro.representations.sum import Sum
from systemds.scuro.representations.zero_pad import ZeroPad

IMPLICIT = AggregatedRepresentation(
    target_dimensions=1, aggregate_leading=True
).get_current_parameters()
FEATURE_AGG = AggregatedRepresentation(target_dimensions=1).get_current_parameters()
NONE = frozenset()


def S(*c, op=Sum):
    return (op, {}, *c)


def A(*c):
    return (Average, {}, *c)


def C(*c):
    return (Concatenation, {"preserve_leading_axis": False}, *c)


def W(*c):
    return (Concatenation, {"preserve_leading_axis": True}, *c)


def L(*c):
    return (LSTM, {}, *c)


def G(c, params=IMPLICIT):
    return (AggregatedRepresentation, params, c)


def P(c):
    return (ZeroPad, ZeroPad(8, 3).get_current_parameters(), c)


def pushed(c, params):
    return (BoW, {"_pushdown_aggregation": params}, c)


# (name, left, right, laws, invariant, equal)
CASES = [
    ("leaf", "a", "a", ALL_LAWS, False, True),
    ("leaf_index", "a", "b", ALL_LAWS, False, False),
    ("params", (BoW, {"x": 1}, "a"), (BoW, {"x": 2}, "a"), ALL_LAWS, False, False),
    *[
        (
            f"comm_{op.__name__}",
            S("a", "b", op=op),
            S("b", "a", op=op),
            ALL_LAWS,
            False,
            True,
        )
        for op in (Sum, RowMax, Hadamard)
    ],
    ("comm_off", S("a", "b"), S("b", "a"), NONE, False, False),
    ("assoc", S(S("a", "b"), "c"), S("a", S("b", "c")), ALL_LAWS, False, True),
    ("assoc_nary", S(S("a", "b"), "c"), S("a", "b", "c"), ALL_LAWS, False, True),
    ("assoc_off", S(S("a", "b"), "c"), S("a", "b", "c"), {"comm"}, False, False),
    (
        "assoc_other_op",
        S(S("a", "b", op=RowMax), "c"),
        S("a", "b", "c"),
        ALL_LAWS,
        False,
        False,
    ),
    ("concat_order_kept", C("a", "b"), C("b", "a"), ALL_LAWS, False, False),
    ("concat_perm", C("a", "b"), C("b", "a"), ALL_LAWS, True, True),
    ("concat_assoc", C(C("a", "b"), "c"), C("a", C("b", "c")), ALL_LAWS, False, True),
    ("window_concat_never_sorted", W("a", "b"), W("b", "a"), ALL_LAWS, True, False),
    (
        "window_concat_assoc",
        W(W("a", "b"), "c"),
        W("a", "b", "c"),
        ALL_LAWS,
        True,
        True,
    ),
    ("window_concat_not_concat", W("a", "b"), C("a", "b"), ALL_LAWS, True, False),
    ("lstm_never_sorted", L("a", "b"), L("b", "a"), ALL_LAWS, True, False),
    (
        "no_comm_below_lstm",
        L(S("a", "b"), "c"),
        L(S("b", "a"), "c"),
        ALL_LAWS,
        True,
        False,
    ),
    (
        "no_perm_below_lstm",
        L(C("a", "b"), "c"),
        L(C("b", "a"), "c"),
        ALL_LAWS,
        True,
        False,
    ),
    (
        "pad_below_lstm",
        L(S(P("a"), "b"), "c"),
        L(S("a", "b"), "c"),
        ALL_LAWS,
        False,
        True,
    ),
    (
        "comm_above_lstm",
        S(L("a", "b"), "c"),
        S("c", L("a", "b")),
        ALL_LAWS,
        False,
        True,
    ),
    ("avg_flatten", A(A("a", "b"), "c"), A("c", A("b", "a")), ALL_LAWS, False, True),
    ("avg_not_assoc", A(A("a", "b"), "c"), A("a", A("b", "c")), ALL_LAWS, False, False),
    (
        "avg_weights_add",
        A(A("a", "b"), A("a", "c")),
        A("a", A("b", "c")),
        ALL_LAWS,
        False,
        True,
    ),
    (
        "avg_flatten_off",
        A(A("a", "b"), A("a", "c")),
        A("a", A("b", "c")),
        {"comm"},
        False,
        False,
    ),
    (
        "agg_pushdown",
        pushed("a", FEATURE_AGG),
        G((BoW, {}, "a"), FEATURE_AGG),
        ALL_LAWS,
        False,
        True,
    ),
    (
        "agg_pushdown_off",
        pushed("a", FEATURE_AGG),
        G((BoW, {}, "a"), FEATURE_AGG),
        NONE,
        False,
        False,
    ),
    ("implicit_agg", S(G("a"), "b"), S("a", "b"), ALL_LAWS, False, True),
    (
        "implicit_agg_pushed",
        S(pushed("a", IMPLICIT), "b"),
        S((BoW, {}, "a"), "b"),
        ALL_LAWS,
        False,
        True,
    ),
    (
        "implicit_agg_off",
        S(G("a"), "b"),
        S("a", "b"),
        ALL_LAWS - {"implicit_agg"},
        False,
        False,
    ),
    (
        "implicit_agg_other_params",
        S(G("a", FEATURE_AGG), "b"),
        S("a", "b"),
        ALL_LAWS,
        False,
        False,
    ),
    ("implicit_agg_window_concat", W(G("a"), "b"), W("a", "b"), ALL_LAWS, False, False),
    ("implicit_agg_lstm", L(G("a"), "b"), L("a", "b"), ALL_LAWS, False, False),
    ("implicit_agg_unary", (BoW, {}, G("a")), (BoW, {}, "a"), ALL_LAWS, False, False),
    ("implicit_pad", S(P("a"), "b"), S("a", "b"), ALL_LAWS, False, True),
    ("implicit_pad_avg", A(P("a"), "b"), A("a", "b"), ALL_LAWS, False, True),
    (
        "implicit_pad_off",
        S(P("a"), "b"),
        S("a", "b"),
        ALL_LAWS - {"implicit_pad"},
        False,
        False,
    ),
    ("implicit_pad_concat", C(P("a"), "b"), C("a", "b"), ALL_LAWS, True, False),
]


def build(expr):
    builder = RepresentationDAGBuilder()
    leaves = {}

    def make(e):
        if isinstance(e, str):
            if e not in leaves:
                i = "abc".index(e)
                leaves[e] = builder.create_leaf_node(f"m{i}", i)
            return leaves[e]
        op, params, *children = e
        return builder.create_operation_node(op, [make(c) for c in children], params)

    return builder.build(make(expr))


class TestCanonicalId(unittest.TestCase):
    def test_canonical_pairs(self):
        for name, left, right, laws, invariant, equal in CASES:
            with self.subTest(name):
                laws = frozenset(laws)
                left, right = build(left), build(right)
                ids = [canonical_id(dag, laws, invariant) for dag in (left, right)]
                self.assertEqual(ids[0] == ids[1], equal)
                for dag, dag_id in zip((left, right), ids):
                    spec = json.loads(json.dumps(dag.to_spec()))
                    self.assertEqual(canonical_id(spec, laws, invariant), dag_id)

    def test_dag_method_defaults(self):
        dag = build(C("a", S("c", "b")))
        self.assertEqual(dag.canonical_id(), canonical_id(dag, ALL_LAWS, False))
        self.assertNotEqual(dag.canonical_id(laws=NONE), dag.canonical_id())

    def test_sub_plan_signature(self):
        dag = build(C("a", S("b", "c")))
        sub = dag.get_node_by_id(dag.get_node_by_id(dag.root_node_id).inputs[1])
        self.assertEqual(
            canonical_id(dag, ALL_LAWS, False, sub.node_id),
            canonical_id(build(S("c", "b")), ALL_LAWS, False),
        )


if __name__ == "__main__":
    unittest.main()

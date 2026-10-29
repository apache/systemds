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
import hashlib
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb, factorial, prod
from typing import NamedTuple, Optional

import numpy as np

from systemds.scuro.drsearch.operator_registry import Registry, load_builtin_operators
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.context import Context
from systemds.scuro.representations.dimensionality_reduction import (
    DimensionalityReduction,
)
from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.representations.representation import (
    FitRequirement,
    RepresentationStats,
)
from systemds.scuro.representations.zero_pad import ZeroPad

ALL_LAWS = frozenset(
    {
        "comm",
        "assoc",
        "avg_flatten",
        "concat_perm",
        "agg_pushdown",
        "implicit_agg",
        "implicit_pad",
    }
)

EXACT_LAWS = frozenset({"agg_pushdown", "implicit_agg", "implicit_pad"})


@dataclass(frozen=True)
class AlgebraConfig:
    """Switches for the Beta-algebra rules; the defaults are the legacy search."""

    laws: frozenset = frozenset()
    feature_order_invariant: Optional[bool] = None
    cse_law_aware: bool = False
    ga_canonical_dedup: bool = False
    exhaustive_dedup_permutations: bool = False
    exhaustive_canonical: bool = False
    typecheck: str = "off"  # off | warn | strict
    explicit_fusion_prep: bool = False
    explicit_padding: str = "off"  # off | pad | reject
    typed_chaining: bool = False
    share_fused_subtrees: bool = False

    @classmethod
    def full(cls) -> "AlgebraConfig":
        return cls(
            laws=ALL_LAWS,
            feature_order_invariant=True,
            cse_law_aware=True,
            ga_canonical_dedup=True,
            exhaustive_dedup_permutations=True,
            exhaustive_canonical=True,
            typecheck="warn",
            explicit_fusion_prep=True,
            explicit_padding="pad",
            share_fused_subtrees=True,
        )


class OperatorSignature(NamedTuple):
    kind: str  # encode | context | aggregate | reduce | fuse | task
    input_types: frozenset
    output_type: Optional[ModalityType]  # None: the input's type
    arity: object  # 1, or "n" for fusion
    input_rank: str  # any | seq | windows | vector
    output_rank: str


ANY_TYPE = frozenset(m for m in ModalityType if isinstance(m.value, int))


def operator_signature(op_cls, parameters=None) -> OperatorSignature:
    """`op_cls` None is a task node."""
    if op_cls is None:
        return OperatorSignature("task", ANY_TYPE, None, 1, "any", "any")
    if issubclass(op_cls, AggregatedRepresentation):
        return OperatorSignature("aggregate", ANY_TYPE, None, 1, "seq", "vector")
    if issubclass(op_cls, ZeroPad):
        return OperatorSignature("reduce", ANY_TYPE, None, 1, "vector", "vector")
    if issubclass(op_cls, Fusion):
        keep_axis = (parameters or {}).get("preserve_leading_axis")
        seq_in = op_cls.preserves_leading_axis if keep_axis is None else keep_axis
        ranks = ["seq" if seq else "vector" for seq in (seq_in, keep_axis)]
        return OperatorSignature("fuse", ANY_TYPE, _output_type(op_cls), "n", *ranks)
    if issubclass(op_cls, Context):
        kind, lists, ranks = "context", Registry._context_operators, ("seq", "windows")
    elif issubclass(op_cls, DimensionalityReduction):
        kind, lists = "reduce", Registry._dimensionality_reduction_operators
        ranks = ("vector", "vector")
    else:
        kind, lists, ranks = "encode", Registry._representations, ("any", "any")
    load_builtin_operators(ANY_TYPE)
    input_types = frozenset(m for m, ops in lists.items() if op_cls in ops)
    return OperatorSignature(kind, input_types, _output_type(op_cls), 1, *ranks)


@lru_cache(maxsize=None)
def _needs_alignment(op_cls) -> bool:
    return op_cls().needs_alignment


@lru_cache(maxsize=None)
def _output_type(op_cls):
    return getattr(op_cls, "OUTPUT_MODALITY_TYPE", None) or getattr(
        op_cls(), "output_modality_type", None
    )


class Violation(NamedTuple):
    severity: str  # error | implicit
    kind: str  # input_type | arity | rank | hidden_aggregation | zero_padding
    node_id: str
    operator: str
    detail: str


def typecheck(dag, leaf_types, leaf_stats=None) -> list:
    nodes = {node.node_id: node for node in dag.nodes}
    types, violations = {}, []
    stats = output_stats(nodes.__getitem__, leaf_stats)

    def key(node):
        return node.modality_id, node.representation_index

    def flag(severity, kind, node, detail):
        name = node.operation.__name__
        violations.append(Violation(severity, kind, node.node_id, name, detail))

    def check_fusion(node):
        operands = [stats(i) for i in node.inputs]
        if None in operands:
            return
        op = node.operation(params=node.parameters)
        if not op.preserves_leading_axis:
            ranks = [len(s.output_shape) for s in operands]
            if max(ranks) > 1:
                flag("implicit", "hidden_aggregation", node, f"operand ranks {ranks}")
        if op.needs_alignment:
            widths = [s.output_shape[-1] for s in op._fusion_input_stats(operands)]
            if len(set(widths)) > 1:
                flag("implicit", "zero_padding", node, f"operand widths {widths}")

    def rank(nid):
        """seq or vector, None if unknown. Raw modalities declare a max_length."""
        node = nodes[nid]
        if node.inputs and node.operation is not None:
            declared = operator_signature(node.operation, node.parameters).output_rank
            if declared != "any":
                return "vector" if declared == "vector" else "seq"
        s = stats(nid)
        if s is None or not s.output_shape:
            return None
        seq = len(s.output_shape) > 1 or getattr(s, "max_length", None) is not None
        return "seq" if seq else "vector"

    def check_rank(node, sig):
        got = rank(node.inputs[0])
        if got is not None and (got == "vector") != (sig.input_rank == "vector"):
            flag("error", "rank", node, f"{got} input, expects {sig.input_rank}")

    def out_type(nid):
        if nid in types:
            return types[nid]
        node = nodes[nid]
        if not node.inputs:
            types[nid] = leaf_types.get(key(node))
            return types[nid]
        in_types = [out_type(i) for i in node.inputs]
        if node.operation is None:
            types[nid] = in_types[0]
            return types[nid]
        sig = operator_signature(node.operation, node.parameters)
        if (sig.arity == "n") != (len(node.inputs) > 1):
            flag("error", "arity", node, f"{len(node.inputs)} input(s)")
        for t in in_types:
            if t is not None and t not in sig.input_types:
                flag("error", "input_type", node, f"{t.name} input")
        if sig.arity == 1 and len(node.inputs) == 1 and sig.input_rank != "any":
            check_rank(node, sig)
        if sig.kind == "fuse" and leaf_stats is not None:
            check_fusion(node)
        types[nid] = sig.output_type or in_types[0]
        return types[nid]

    out_type(dag.root_node_id)
    return violations


def output_stats(get_node, leaf_stats):
    from systemds.scuro.drsearch.representation_dag import (
        pushdown_aggregation_for_node,
    )

    memo = {}

    def stats(nid):
        if nid not in memo:
            node = get_node(nid)
            memo[nid] = None
            if not node.inputs:
                key = node.modality_id, node.representation_index
                memo[nid] = (leaf_stats or {}).get(key)
            elif node.operation is not None:
                inputs = [stats(i) for i in node.inputs]
                if None not in inputs:
                    try:
                        out = node.operation(params=node.parameters).get_output_stats(
                            inputs if len(inputs) > 1 else inputs[0]
                        )
                        agg = pushdown_aggregation_for_node(node.parameters)
                        memo[nid] = agg.get_output_stats(out) if agg else out
                    except Exception:
                        pass
        return memo[nid]

    return stats


def sample_stats(modality) -> Optional[RepresentationStats]:
    data = getattr(modality, "data", None)
    if data is None or len(data) == 0:
        return None
    try:
        sample = np.asarray(data[0])
    except (TypeError, ValueError):
        return None
    if sample.dtype == object:
        return None
    shape = tuple(sample.shape) if sample.ndim else (1,)
    return RepresentationStats(len(data), shape, output_shape_is_known=True)


def explicit_prep_on(algebra) -> bool:
    return algebra.explicit_fusion_prep or algebra.explicit_padding == "pad"


def explicit_fusion_operands(
    builder, operation, parameters, inputs, leaf_stats, algebra
) -> list:
    if not issubclass(operation, Fusion):
        return inputs
    op = operation(params=parameters)
    stats = output_stats(builder.get_node, leaf_stats)
    if algebra.explicit_fusion_prep and not op.preserves_leading_axis:
        agg = AggregatedRepresentation(target_dimensions=1, aggregate_leading=True)
        inputs = [
            (
                builder.create_operation_node(
                    AggregatedRepresentation, [i], agg.get_current_parameters()
                )
                if stats(i) is not None and len(stats(i).output_shape) > 1
                else i
            )
            for i in inputs
        ]
    shapes = [stats(i) for i in inputs]
    if (
        algebra.explicit_padding == "pad"
        and op.needs_alignment
        and None not in shapes
        and all(len(s.output_shape) == 1 for s in shapes)
    ):
        width = max(s.output_shape[0] for s in shapes)
        inputs = [
            (
                builder.create_operation_node(
                    ZeroPad,
                    [i],
                    ZeroPad(width, s.output_shape[0]).get_current_parameters(),
                )
                if s.output_shape[0] < width
                else i
            )
            for i, s in zip(inputs, shapes)
        ]
    return inputs


def count_explicit_prep(counter, dag) -> None:
    """Counts the explicit operand preparation in a plan about to be evaluated."""
    nodes = {n.node_id: n for n in dag.filter_connected_nodes(dag.nodes)}
    padded = False
    for node in nodes.values():
        if not (node.operation and issubclass(node.operation, Fusion)):
            continue
        for operand in (nodes[i] for i in node.inputs):
            if operand.operation is ZeroPad:
                padded = True
                ratio = operand.parameters["width"] / operand.parameters["input_width"]
                counter["explicit_paddings"] += 1
                counter["explicit_padding_ratio_sum"] += ratio
                counter["explicit_padding_ratio_max"] = max(
                    counter["explicit_padding_ratio_max"], ratio
                )
                operand = nodes[operand.inputs[0]]
            if operand.operation is AggregatedRepresentation:
                counter["explicit_aggregations"] += 1
    counter["explicit_prep_plans"] += 1
    counter["explicit_padding_plans"] += padded


def count_violations(counter, violations) -> list:
    """Adds per-kind and per-operator counts; returns the errors."""
    counter["typecheck_plans"] += 1
    for v in violations:
        counter[f"typecheck_{v.kind}"] += 1
        counter[f"typecheck_{v.kind}:{v.operator}"] += 1
    return [v for v in violations if v.severity == "error"]


def order_invariant(config: AlgebraConfig, tasks) -> bool:
    if config.feature_order_invariant is not None:
        return config.feature_order_invariant
    return bool(tasks) and all(
        getattr(task.model, "feature_order_invariant", None) is True for task in tasks
    )


def canonical_signature(dag, laws, invariant, node_id=None) -> tuple:
    # representation_dag imports this module.
    from systemds.scuro.drsearch.representation_dag import _spec_safe

    def view(nid):
        node = dag.get_node_by_id(nid)
        op = node.operation
        params = {k: _spec_safe(v) for k, v in (node.parameters or {}).items()}
        name = op.__name__ if op is not None else None
        return (
            name,
            op,
            params,
            node.inputs,
            node.modality_id,
            node.representation_index,
        )

    return _signature(view, node_id or dag.root_node_id, laws, invariant)


def canonical_signature_from_spec(spec: dict, laws, invariant) -> tuple:
    nodes = {node["id"]: node for node in spec["nodes"]}
    classes = _fusion_classes()

    def view(nid):
        n = nodes[nid]
        return (
            n["op"],
            classes.get(n["op"]),
            n["params"],
            n["inputs"],
            n["modality_id"],
            n["representation_index"],
        )

    return _signature(view, spec["root"], laws, invariant)


def canonical_id(dag, laws, invariant, node_id=None) -> str:
    """`dag` is a RepresentationDag or a to_spec() dict."""
    if isinstance(dag, dict):
        sig = canonical_signature_from_spec(dag, laws, invariant)
    else:
        sig = canonical_signature(dag, laws, invariant, node_id)
    blob = json.dumps(sig, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _fusion_classes() -> dict:
    classes = {cls.__name__: cls for cls in Registry().get_fusion_operators()}
    stack = [Fusion]
    while stack:
        cls = stack.pop()
        classes.setdefault(cls.__name__, cls)
        stack.extend(cls.__subclasses__())
    return classes


def _params_key(params) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _signature(view, root, laws, invariant) -> tuple:
    # The aggregation Fusion._prepare_modalities applies to every operand of a
    # fusion that does not preserve the leading axis.
    implicit_agg = (
        "op",
        AggregatedRepresentation.__name__,
        _params_key(
            AggregatedRepresentation(
                target_dimensions=1, aggregate_leading=True
            ).get_current_parameters()
        ),
    )
    memo = {}

    def sig(nid, laws):
        if (nid, laws) not in memo:
            memo[nid, laws] = compute(nid, laws)
        return memo[nid, laws]

    def compute(nid, laws):
        name, op, params, inputs, modality_id, representation_index = view(nid)
        if not inputs:
            return ("leaf", modality_id, representation_index)

        below = laws & EXACT_LAWS if _fitted(op) else laws - {"concat_perm"}
        children = [sig(i, below) for i in inputs]
        params = dict(params)
        pushed = (
            params.pop("_pushdown_aggregation", None)
            if "agg_pushdown" in laws
            else None
        )
        head = ("op", name, _params_key(params))
        fusion = isinstance(op, type) and issubclass(op, Fusion)
        op_laws = op.algebraic_laws(params) & laws if fusion else frozenset()

        if (
            fusion
            and "implicit_agg" in laws
            and not params.get("preserve_leading_axis", op.preserves_leading_axis)
        ):
            children = [c[3][0] if c[:3] == implicit_agg else c for c in children]
        if fusion and "implicit_pad" in laws and _needs_alignment(op):
            children = [
                c[3][0] if c[:2] == ("op", ZeroPad.__name__) else c for c in children
            ]
        if "assoc" in op_laws:
            children = [g for c in children for g in (c[3] if c[:3] == head else (c,))]
        if "avg_flatten" in op_laws:
            weights = Counter()
            for c in children:
                for w, g in c[3] if c[:3] == head else (("1", c),):
                    weights[g] += Fraction(w) / len(children)
            children = sorted(((str(w), g) for g, w in weights.items()), key=repr)
        elif "comm" in op_laws or ("concat_perm" in op_laws and invariant):
            children = sorted(children, key=repr)

        out = head + (tuple(children),)
        if pushed is not None:
            out = ("op", AggregatedRepresentation.__name__, _params_key(pushed), (out,))
        return out

    return sig(root, frozenset(laws))


def count_fusion_plans(n, fusion_ops, laws, invariant) -> dict:
    heads = _fusion_heads(fusion_ops, laws)
    f = len(heads)
    reorders = sum(_reorders(op_laws, False) for *_, op_laws in heads)
    ordered = factorial(2 * n - 2) // factorial(n - 1) * f ** (n - 1)
    if n == 1:
        canonical = 1
    elif reorders == f and all("assoc" in op_laws for *_, op_laws in heads):
        canonical = f * _flattened_trees(n, f)
    elif n <= 4:
        canonical = _brute_force_canonical(n, fusion_ops, laws, invariant)
    else:
        canonical = None
    return {
        "ordered": ordered,
        "legacy_exhaustive": factorial(n) * ordered,
        "commutative": _commutative_plans(n, heads, invariant),
        "canonical": canonical,
    }


def count_search_space(
    ks, fusion_ops, laws, invariant, min_modalities=2, max_modalities=None
) -> dict:
    """`ks[i]` is the number of representations of modality i."""
    totals = dict.fromkeys(
        ("ordered", "legacy_exhaustive", "commutative", "canonical"), 0
    )
    for r in range(min_modalities, (max_modalities or len(ks)) + 1):
        leaves = sum(prod(c) for c in itertools.combinations(ks, r))
        for key, plans in count_fusion_plans(r, fusion_ops, laws, invariant).items():
            if plans is None or totals[key] is None:
                totals[key] = None
            else:
                totals[key] += leaves * plans
    return totals


def _fitted(op) -> bool:
    return getattr(op, "fit_requirement", None) not in (None, FitRequirement.NONE)


def _commutative_plans(n, heads, invariant) -> int:
    plans = {(1, True): 1, (1, False): 1}

    def count(m, free, invariant):
        total = 0
        for *_, op, _, op_laws in heads:
            below = free and not _fitted(op)
            splits = sum(
                comb(m, k) * plans[k, below] * plans[m - k, below] for k in range(1, m)
            )
            total += splits // (2 if free and _reorders(op_laws, invariant) else 1)
        return total

    for m in range(2, n + 1):
        for free in (True, False):
            plans[m, free] = count(m, free, False)
    return count(n, True, invariant) if n > 1 else 1


def _reorders(op_laws, invariant) -> bool:
    return "comm" in op_laws or ("concat_perm" in op_laws and invariant)


def _fusion_heads(fusion_ops, laws) -> list:
    from systemds.scuro.drsearch.representation_dag import _spec_safe

    heads = []
    for op in fusion_ops:
        params = {k: _spec_safe(v) for k, v in op().get_current_parameters().items()}
        heads.append((op.__name__, op, params, op.algebraic_laws(params) & laws))
    return heads


def _flattened_trees(n, f) -> int:
    tree, block, partitions = [0, 1], [0, 1], [1, 1]
    for m in range(2, n + 1):
        tree.append(
            sum(comb(m - 1, j - 1) * block[j] * partitions[m - j] for j in range(1, m))
        )
        block.append((f - 1) * tree[m])
        partitions.append(tree[m] + block[m])
    return tree[n]


def _ordered_trees(leaves):
    if len(leaves) == 1:
        yield leaves[0]
        return
    for split in range(1, len(leaves)):
        for left in itertools.combinations(leaves, split):
            right = tuple(i for i in leaves if i not in left)
            for l_tree in _ordered_trees(left):
                for r_tree in _ordered_trees(right):
                    yield (l_tree, r_tree)


def _brute_force_canonical(n, fusion_ops, laws, invariant) -> int:
    heads = _fusion_heads(fusion_ops, laws)
    signatures = set()
    for tree in _ordered_trees(tuple(range(n))):
        for choice in itertools.product(heads, repeat=n - 1):
            nodes, ops = {}, iter(choice)

            def add(t):
                if isinstance(t, int):
                    nodes[t] = (None, None, {}, (), t, 0)
                    return t
                inputs = (add(t[0]), add(t[1]))
                name, op, params, _ = next(ops)
                nid = f"n{len(nodes)}"
                nodes[nid] = (name, op, params, inputs, None, None)
                return nid

            root = add(tree)
            signatures.add(_signature(nodes.__getitem__, root, laws, invariant))
    return len(signatures)

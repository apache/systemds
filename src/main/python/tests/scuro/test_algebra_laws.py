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

import itertools
import unittest

import numpy as np

from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.concatenation import Concatenation

OPERATORS = [(op, {}) for op in Registry().get_fusion_operators()] + [
    (Concatenation, {"preserve_leading_axis": True})
]
DECLARED = [
    (op, params, law)
    for op, params in OPERATORS
    for law in sorted(op.algebraic_laws(params))
]
WIDTHS = {"equal": (6, 6, 6), "unequal": (6, 3, 9)}
DTYPES = {
    "float32": (np.float32, np.float32, np.float32),
    "float32/64": (np.float32, np.float64, np.float32),
    "int/float": (np.int64, np.float32, np.float64),
}


def _operands(widths, dtypes, rank):
    rng = np.random.default_rng(0)
    shape = (5,) + (4,) * (rank - 1)
    return [
        (rng.standard_normal(shape + (w,)) * 5).astype(dtype)
        for w, dtype in zip(widths, dtypes)
    ]


def _fuse(op, params, *operands):
    modalities = []
    for data in operands:
        metadata = [ModalityType.EMBEDDING.create_metadata(d) for d in data]
        modality = TransformedModality(
            Modality(ModalityType.EMBEDDING, 0, metadata), "operand"
        )
        modality.data = data
        modalities.append(modality)
    return np.asarray(op(params).transform(modalities))


def _assert_same(x, y, exact):
    assert x.shape == y.shape
    if exact:
        np.testing.assert_array_equal(x, y)
    else:
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)


def _prepared(x, width):
    # What _prepare_modalities hands to execute: hidden mean aggregation, then zero-padding.
    x = x.mean(axis=1) if x.ndim == 3 else x
    return np.pad(x, ((0, 0), (0, width - x.shape[-1])))


def _comm(f, a, b, c, exact):
    _assert_same(f(a, b), f(b, a), exact)


def _assoc(f, a, b, c, exact):
    left = f(f(a, b), c)
    _assert_same(left, f(a, f(b, c)), exact)
    _assert_same(left, f(a, b, c), exact)


def _avg_flatten(f, a, b, c, exact):
    width = max(x.shape[-1] for x in (a, b, c))
    expected = sum(w * _prepared(x, width) for w, x in ((0.25, a), (0.25, b), (0.5, c)))
    _assert_same(f(f(a, b), c), expected, exact)


def _concat_perm(f, a, b, c, exact):
    _assert_same(f(a, b), np.roll(f(b, a), -b.shape[-1], axis=-1), exact=True)


LAW_CHECKS = {
    "comm": _comm,
    "assoc": _assoc,
    "avg_flatten": _avg_flatten,
    "concat_perm": _concat_perm,
}


class TestAlgebraLaws(unittest.TestCase):
    def test_every_declared_law_is_checked(self):
        self.assertLessEqual({law for _, _, law in DECLARED}, set(LAW_CHECKS))

    def test_declared_law_holds(self):
        cases = itertools.product(DECLARED, WIDTHS, DTYPES, (1, 2))
        for (op, params, law), widths, dtypes, rank in cases:
            with self.subTest(
                op=f"{op.__name__}{params or ''}",
                law=law,
                widths=widths,
                dtypes=dtypes,
                rank=rank,
            ):
                LAW_CHECKS[law](
                    lambda *xs: _fuse(op, params, *xs),
                    *_operands(WIDTHS[widths], DTYPES[dtypes], rank),
                    exact=op is Concatenation,
                )

    def test_window_axis_concatenation_is_order_sensitive(self):
        params = {"preserve_leading_axis": True}
        a, b, _ = _operands(WIDTHS["equal"], DTYPES["float32"], rank=2)
        self.assertFalse(
            np.array_equal(
                _fuse(Concatenation, params, a, b), _fuse(Concatenation, params, b, a)
            )
        )
        self.assertFalse({"comm", "concat_perm"} & Concatenation.algebraic_laws(params))


if __name__ == "__main__":
    unittest.main()

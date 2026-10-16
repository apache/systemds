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

import unittest
import math

import numpy as np

from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import RepresentationStats
from tests.scuro.data_generator import ModalityRandomDataGenerator, TestDataLoader
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.representations.timeseries_representations import (
    FrequencyMagnitude,
    Mean,
    Quantile,
    Std,
)
from systemds.scuro.representations.physiological_window import (
    AdaptiveWindow,
    PhysiologicalEventWindow,
)
from systemds.scuro.representations.window_aggregation import (
    StaticWindow,
    DynamicWindow,
    WindowAggregation,
    resolve_aggregation_function,
)


class _FakeModality:
    """The smallest surface a context operator's execute() actually touches.

    Lets a test hand a window operator hand-built instances -- an empty one, a
    flat one -- without going through a loader that would reject them first.
    """

    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata or [
            ModalityType.TIMESERIES.create_metadata(["signal"], np.asarray(instance))
            for instance in data
        ]

    def get_data_layout(self):
        from systemds.scuro.modality.type import DataLayout

        return DataLayout.SINGLE_LEVEL


class TestWindowOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_instances = 4
        cls.data_generator = ModalityRandomDataGenerator()
        cls.aggregations = ["mean", "sum", "max", "min"]

    def test_static_window(self):
        num_windows = 5
        data, md = self.data_generator.create_visual_modality(self.num_instances, 10)
        modality = UnimodalModality(
            TestDataLoader(
                [i for i in range(0, self.num_instances)],
                None,
                ModalityType.VIDEO,
                data,
                np.float32,
                md,
            )
        )
        aggregated_window = modality.context(StaticWindow(num_windows=num_windows))

        for i in range(0, self.num_instances):
            assert len(aggregated_window.data[i]) == num_windows

    def test_dynamic_window(self):
        num_windows = 5
        data, md = self.data_generator.create_visual_modality(self.num_instances, 10)
        modality = UnimodalModality(
            TestDataLoader(
                [i for i in range(0, self.num_instances)],
                None,
                ModalityType.VIDEO,
                data,
                np.float32,
                md,
            )
        )
        aggregated_window = modality.context(DynamicWindow(num_windows=num_windows))

        for i in range(0, self.num_instances):
            assert len(aggregated_window.data[i]) == num_windows

    def test_window_aggregation_on_1d_modalities(self):
        # create1DModality returns the same shape and dtype for all three
        # modality types. window_aggregation looks at the data layout and not
        # at the modality type, so the result should be the same for all of
        # them.
        window_size = 10

        for modality_type in [
            ModalityType.AUDIO,
            ModalityType.VIDEO,
            ModalityType.TEXT,
        ]:
            r = self.data_generator.create1DModality(
                self.num_instances, 200, modality_type
            )
            for aggregation in self.aggregations:
                with self.subTest(modality=modality_type.name, aggregation=aggregation):
                    windowed_modality = r.window_aggregation(window_size, aggregation)
                    self.verify_window_operation(
                        aggregation, r, windowed_modality, window_size
                    )

    def test_window_aggregation_on_nd_modality(self):
        # Window aggregation only changes the first (time) axis and keeps the
        # feature axes as they are. The expected shape is therefore
        # (num_windows,) + dims[1:] for any number of dimensions.
        num_windows = 10

        for dims in [(100, 8, 8), (100, 8)]:
            if len(dims) == 3:
                data, _ = self.data_generator.create_3d_modality(
                    self.num_instances, dims
                )
            else:
                data, _ = self.data_generator.create_2d_modality(
                    self.num_instances, dims
                )
            embedding_modality = TransformedModality(
                self.data_generator, "test_transformation"
            )
            embedding_modality.data = data
            embedding_modality.stats = RepresentationStats(self.num_instances, dims)

            for window_operator in [
                StaticWindow(num_windows=num_windows),
                DynamicWindow(num_windows=num_windows),
                WindowAggregation(window_size=10),
            ]:
                with self.subTest(dims=dims, operator=type(window_operator).__name__):
                    stats = window_operator.get_output_stats(embedding_modality.stats)
                    self.assertEqual(stats.num_instances, self.num_instances)
                    self.assertEqual(stats.output_shape, (num_windows,) + dims[1:])

                    embedding_modality.context(window_operator)

    def _timeseries_modality(self, signal_length=100):
        return self.data_generator.create1DModality(
            self.num_instances, signal_length, ModalityType.TIMESERIES
        )

    # ------------------------------------------------------------------
    # WindowAggregation: window size against signal length
    # ------------------------------------------------------------------

    def test_window_size_of_one_leaves_the_signal_unchanged(self):
        signal_length = 60
        modality = self._timeseries_modality(signal_length)
        windowed = np.asarray(
            WindowAggregation("mean", window_size=1).execute(modality)
        )
        self.assertEqual(windowed.shape, (self.num_instances, signal_length))
        np.testing.assert_allclose(windowed, modality.data, rtol=1e-5, atol=1e-5)

    def test_window_larger_than_the_signal_collapses_to_one_padded_window(self):
        """The single short window is zero-padded up to window_size, so a mean
        over it divides by the nominal size and not by the sample count. That
        dilution is the behaviour a caller has to be able to rely on."""
        signal_length = 100
        window_size = 250
        modality = self._timeseries_modality(signal_length)

        windowed = np.asarray(
            WindowAggregation("mean", window_size=window_size).execute(modality)
        )
        self.assertEqual(windowed.shape, (self.num_instances, 1))
        np.testing.assert_allclose(
            windowed[:, 0],
            modality.data.sum(axis=1) / window_size,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_window_size_that_does_not_divide_the_signal_keeps_a_tail_window(self):
        signal_length = 100
        window_size = 7
        modality = self._timeseries_modality(signal_length)

        windowed = np.asarray(
            WindowAggregation("mean", window_size=window_size).execute(modality)
        )
        self.assertEqual(
            windowed.shape,
            (self.num_instances, math.ceil(signal_length / window_size)),
        )
        # The tail window covers only the samples that are actually there.
        tail_start = (windowed.shape[1] - 1) * window_size
        np.testing.assert_allclose(
            windowed[:, -1],
            modality.data[:, tail_start:].mean(axis=1),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_batched_and_per_instance_paths_agree(self):
        """Equal-length numeric instances take a vectorized path; anything else
        falls back to the per-instance loop. A window's value must not depend on
        which of the two ran."""
        modality = self._timeseries_modality(300)
        for window_size in [1, 7, 10, 300, 400]:
            with self.subTest(window_size=window_size):
                batched = np.asarray(
                    WindowAggregation("mean", window_size=window_size).execute(modality)
                )
                per_instance_operator = WindowAggregation(
                    "mean", window_size=window_size
                )
                per_instance = np.stack(
                    [
                        per_instance_operator.window_aggregate_single_level(
                            np.asarray(instance),
                            math.ceil(len(instance) / window_size),
                        )
                        for instance in modality.data
                    ]
                )
                np.testing.assert_allclose(batched, per_instance, rtol=1e-5, atol=1e-5)

    def test_window_aggregation_without_padding_returns_one_array_per_instance(self):
        modality = self._timeseries_modality(100)
        windowed = WindowAggregation("mean", window_size=10, pad=False).execute(
            modality
        )
        self.assertIsInstance(windowed, list)
        self.assertEqual(len(windowed), self.num_instances)
        for instance in windowed:
            self.assertEqual(np.asarray(instance).shape, (10,))

    # ------------------------------------------------------------------
    # Preconditions
    # ------------------------------------------------------------------

    def test_empty_instance_is_rejected(self):
        """An empty instance has no window to reduce. Failing loudly beats
        returning an empty feature the model would only choke on later."""
        empty = _FakeModality(
            [np.array([], dtype=np.float32)],
            [
                ModalityType.TIMESERIES.create_metadata(
                    ["signal"], np.zeros(1, dtype=np.float32)
                )
            ],
        )
        with self.assertRaises(ValueError):
            WindowAggregation("mean", window_size=10).execute(empty)

    def test_invalid_aggregation_function_is_rejected(self):
        for operator in [WindowAggregation, StaticWindow, DynamicWindow]:
            with self.subTest(operator=operator.__name__):
                with self.assertRaises(ValueError):
                    operator(aggregation_function=object())
        # Aggregation itself only knows a fixed set of names.
        with self.assertRaises(ValueError):
            Aggregation("not_an_aggregation")

    # ------------------------------------------------------------------
    # StaticWindow / DynamicWindow: window count against signal length
    # ------------------------------------------------------------------

    def test_single_window_reduces_the_whole_signal(self):
        modality = self._timeseries_modality(100)
        for operator_class in [StaticWindow, DynamicWindow]:
            with self.subTest(operator=operator_class.__name__):
                windowed = np.asarray(
                    operator_class("mean", num_windows=1).execute(modality)
                )
                self.assertEqual(windowed.shape, (self.num_instances, 1))
                np.testing.assert_allclose(
                    windowed[:, 0], modality.data.mean(axis=1), rtol=1e-5, atol=1e-5
                )

    def test_static_window_pads_beyond_the_signal_length(self):
        """StaticWindow honours num_windows literally: asking for more windows
        than there are samples zero-pads rather than clamping."""
        signal_length = 100
        num_windows = 250
        modality = self._timeseries_modality(signal_length)

        operator = StaticWindow("mean", num_windows=num_windows)
        windowed = np.asarray(operator.execute(modality))
        self.assertEqual(windowed.shape, (self.num_instances, num_windows))
        self.assertEqual(
            tuple(
                operator.get_output_stats(
                    RepresentationStats(self.num_instances, (signal_length,))
                ).output_shape
            ),
            (num_windows,),
        )

    def test_dynamic_window_clamps_num_windows_to_the_signal_length(self):
        """DynamicWindow splits the signal into geometrically growing windows,
        which cannot be shorter than one sample -- so the count is capped."""
        signal_length = 100
        modality = self._timeseries_modality(signal_length)

        operator = DynamicWindow("mean", num_windows=250)
        windowed = np.asarray(operator.execute(modality))
        self.assertEqual(windowed.shape, (self.num_instances, signal_length))
        self.assertEqual(
            tuple(
                operator.get_output_stats(
                    RepresentationStats(self.num_instances, (signal_length,))
                ).output_shape
            ),
            (signal_length,),
        )

    def test_window_operators_on_variable_length_instances(self):
        """A window *count* is length-independent, so ragged instances still
        stack into one rectangular block."""
        num_windows = 5
        instances = [
            np.random.rand(100 + 37 * i).astype(np.float32)
            for i in range(self.num_instances)
        ]
        modality = _FakeModality(instances)

        for operator_class in [StaticWindow, DynamicWindow]:
            with self.subTest(operator=operator_class.__name__):
                windowed = np.asarray(
                    operator_class("mean", num_windows=num_windows).execute(modality)
                )
                self.assertEqual(windowed.shape, (self.num_instances, num_windows))
                self.assertTrue(np.isfinite(windowed).all())

    # ------------------------------------------------------------------
    # Aggregation functions
    # ------------------------------------------------------------------

    def test_window_operators_accept_a_representation_as_aggregation(self):
        """A window may reduce with a full representation, not just a named
        aggregation -- and then the per-window feature can be multi-valued, so
        get_output_stats has to carry that extra shape."""
        signal_length = 100
        modality = self._timeseries_modality(signal_length)
        input_stats = RepresentationStats(self.num_instances, (signal_length,))

        window_size = 10
        for aggregation, expected_feature_shape in [
            (Mean(), ()),
            (Std(), ()),
            (Quantile(), ()),
            (FrequencyMagnitude(), (window_size // 2 + 1,)),
        ]:
            with self.subTest(aggregation=aggregation.name):
                operator = WindowAggregation(aggregation, window_size=window_size)
                windowed = np.asarray(operator.execute(modality))
                stats = operator.get_output_stats(input_stats)

                expected = (
                    signal_length // window_size,
                    *expected_feature_shape,
                )
                self.assertEqual(windowed.shape, (self.num_instances, *expected))
                self.assertEqual(tuple(stats.output_shape), expected)

        for operator_class in [StaticWindow, DynamicWindow]:
            with self.subTest(operator=operator_class.__name__):
                operator = operator_class(Std(), num_windows=5)
                windowed = np.asarray(operator.execute(modality))
                self.assertEqual(windowed.shape, (self.num_instances, 5))

    def test_window_operator_current_parameters_expose_the_nested_aggregation(self):
        """The tuner reads its search space from get_current_parameters, so a
        representation used as an aggregation has to surface its own parameters
        under a prefixed name rather than disappearing behind the class."""
        window = WindowAggregation("mean", window_size=16)
        parameters = window.get_current_parameters()
        self.assertEqual(parameters["window_size"], 16)
        self.assertIs(parameters["aggregation_function"], Aggregation)
        self.assertEqual(
            parameters["aggregation_function_aggregation_function"], "mean"
        )

        nested = WindowAggregation(Quantile(quantile=0.5), window_size=16)
        nested_parameters = nested.get_current_parameters()
        self.assertIs(nested_parameters["aggregation_function"], Quantile)
        self.assertEqual(nested_parameters["aggregation_function_quantile"], 0.5)

        static = StaticWindow("max", num_windows=7)
        self.assertEqual(static.get_current_parameters()["num_windows"], 7)

    def test_resolve_aggregation_function(self):
        self.assertEqual(resolve_aggregation_function("mean", None), "mean")
        self.assertEqual(
            resolve_aggregation_function("mean", {"aggregation_function": "max"}), "max"
        )
        # A class is instantiated ...
        self.assertIsInstance(
            resolve_aggregation_function("mean", {"aggregation_function": Mean}), Mean
        )
        # ... and its prefixed parameters are threaded into the instance.
        resolved = resolve_aggregation_function(
            "mean",
            {
                "aggregation_function": Quantile,
                "aggregation_function_quantile": 0.25,
            },
        )
        self.assertIsInstance(resolved, Quantile)
        self.assertEqual(resolved.quantile, 0.25)

    # ------------------------------------------------------------------
    # Data-dependent windows
    # ------------------------------------------------------------------

    def test_adaptive_window_output_shape_is_only_an_estimate(self):
        """The window count depends on the signal's local variance, so it is
        not knowable from statistics. The operator must say so rather than
        report a shape the executor would then assert against."""
        signal_length = 300
        modality = self._timeseries_modality(signal_length)
        operator = AdaptiveWindow(
            "mean", base_window_size=64, overlap=0.5, min_window_size=16
        )

        stats = operator.get_output_stats(
            RepresentationStats(self.num_instances, (signal_length,))
        )
        self.assertFalse(stats.output_shape_is_known)
        self.assertEqual(stats.num_instances, self.num_instances)

        windowed = np.asarray(operator.execute(modality))
        self.assertEqual(windowed.shape[0], self.num_instances)
        self.assertGreater(windowed.shape[1], 0)
        self.assertTrue(np.isfinite(windowed).all())

    def test_adaptive_window_clamps_a_floor_above_the_nominal_size(self):
        """A minimum larger than the nominal window is contradictory; it
        collapses onto the nominal size instead of silently inverting."""
        operator = AdaptiveWindow(
            "mean", base_window_size=8, overlap=0.5, min_window_size=64
        )
        self.assertEqual(operator.base_window_size, 8)
        self.assertEqual(operator.min_window_size, 8)

    def test_adaptive_window_always_advances(self):
        """With a small window and a high overlap the stride truncates to zero
        samples, which would never move the cursor. The floor of one sample is
        what keeps execute() from spinning forever."""
        modality = _FakeModality(
            [np.ones(200, dtype=np.float32) for _ in range(self.num_instances)]
        )
        for base_window_size, overlap in [(8, 1.0), (1, 0.9), (4, 0.99)]:
            with self.subTest(base_window_size=base_window_size, overlap=overlap):
                windowed = np.asarray(
                    AdaptiveWindow(
                        "mean",
                        base_window_size=base_window_size,
                        overlap=overlap,
                        min_window_size=1,
                    ).execute(modality)
                )
                self.assertEqual(windowed.shape[0], self.num_instances)
                self.assertGreater(windowed.shape[1], 0)

    def test_physiological_event_window_splits_on_detected_events(self):
        signal_length = 300
        modality = self._timeseries_modality(signal_length)
        operator = PhysiologicalEventWindow(
            "mean", event_threshold=0.5, min_distance=32
        )

        stats = operator.get_output_stats(
            RepresentationStats(self.num_instances, (signal_length,))
        )
        self.assertFalse(stats.output_shape_is_known)

        windowed = np.asarray(operator.execute(modality))
        self.assertEqual(windowed.shape[0], self.num_instances)
        self.assertGreater(windowed.shape[1], 0)
        self.assertTrue(np.isfinite(windowed).all())

    def test_physiological_event_window_falls_back_when_no_event_is_found(self):
        """A flat signal has no peaks to split on, so the operator falls back to
        an even split instead of producing zero windows."""
        min_distance = 32
        for name, instance in [
            ("constant", np.ones(200, dtype=np.float32)),
            ("zeros", np.zeros(200, dtype=np.float32)),
        ]:
            with self.subTest(signal=name):
                modality = _FakeModality(
                    [instance.copy() for _ in range(self.num_instances)]
                )
                windowed = np.asarray(
                    PhysiologicalEventWindow(
                        "mean", event_threshold=0.5, min_distance=min_distance
                    ).execute(modality)
                )
                self.assertEqual(
                    windowed.shape,
                    (self.num_instances, len(instance) // min_distance),
                )
                self.assertTrue(np.isfinite(windowed).all())

    def test_physiological_event_window_floors_min_distance(self):
        self.assertEqual(
            PhysiologicalEventWindow("mean", min_distance=0).min_distance, 1
        )

    def verify_window_operation(
        self, aggregation, modality, windowed_modality, window_size
    ):
        assert windowed_modality.data is not None
        assert len(windowed_modality.data) == self.num_instances

        for i, instance in enumerate(windowed_modality.data):
            # assert (
            #     list(windowed_modality.metadata.values())[i]["data_layout"]["shape"][0]
            #     == list(modality.metadata.values())[i]["data_layout"]["shape"][0]
            # )
            assert len(instance) == math.ceil(len(modality.data[i]) / window_size)
            for j in range(0, len(instance)):
                if aggregation == "mean":
                    np.testing.assert_almost_equal(
                        instance[j],
                        np.mean(
                            modality.data[i][j * window_size : (j + 1) * window_size],
                            axis=0,
                        ),
                    )
                elif aggregation == "sum":
                    np.testing.assert_almost_equal(
                        instance[j],
                        np.sum(
                            modality.data[i][j * window_size : (j + 1) * window_size],
                            axis=0,
                        ),
                    )
                elif aggregation == "max":
                    np.testing.assert_almost_equal(
                        instance[j],
                        np.max(
                            modality.data[i][j * window_size : (j + 1) * window_size],
                            axis=0,
                        ),
                    )
                elif aggregation == "min":
                    np.testing.assert_almost_equal(
                        instance[j],
                        np.min(
                            modality.data[i][j * window_size : (j + 1) * window_size],
                            axis=0,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.test.integration.functions.paramserv;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.paramserv.DataPartitionLocalScheme;
import org.apache.sysml.runtime.controlprogram.paramserv.DataPartitionerLocal;
import org.apache.sysml.runtime.util.DataConverter;
import org.junit.Assert;
import org.junit.Test;

public class DataPartitionerTest {

	@Test
	public void testDataPartitionerDC() {
		DataPartitionerLocal dp = new DataPartitionerLocal(Statement.PSScheme.DISJOINT_CONTIGUOUS);
		double[] df = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		DataPartitionLocalScheme.Result result = dp.doPartitioning(3, DataConverter.convertToMatrixBlock(df, true),
			DataConverter.convertToMatrixBlock(df, true));

		Assert.assertEquals(3, result.pFeatures.size());
		Assert.assertEquals(3, result.pLabels.size());

		double[] expected1 = new double[] { 1, 2, 3, 4 };
		assertResult(result, 0, expected1);

		double[] expected2 = new double[] { 5, 6, 7, 8 };
		assertResult(result, 1, expected2);

		double[] expected3 = new double[] { 9, 10 };
		assertResult(result, 2, expected3);
	}

	private void assertResult(DataPartitionLocalScheme.Result result, int index, double[] expected) {
		List<MatrixObject> pfs = result.pFeatures;
		List<MatrixObject> pls = result.pLabels;
		double[] realValue1 = pfs.get(index).acquireRead().getDenseBlockValues();
		double[] realValue2 = pls.get(index).acquireRead().getDenseBlockValues();
		Assert.assertArrayEquals(expected, realValue1, 0);
		Assert.assertArrayEquals(expected, realValue2, 0);
	}

	@Test
	public void testDataPartitionerDR() {
		DataPartitionerLocal dp = new DataPartitionerLocal(Statement.PSScheme.DISJOINT_RANDOM);
		double[] df = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		DataPartitionLocalScheme.Result result = dp.doPartitioning(4, DataConverter.convertToMatrixBlock(df, true),
			DataConverter.convertToMatrixBlock(df, true));

		Assert.assertEquals(4, result.pFeatures.size());
		Assert.assertEquals(4, result.pLabels.size());

		// Ensure that the index is accorded between features and labels
		IntStream.range(0, result.pFeatures.size()).forEach(i -> {
			double[] f = result.pFeatures.get(i).acquireRead().getDenseBlockValues();
			double[] l = result.pLabels.get(i).acquireRead().getDenseBlockValues();
			Assert.assertArrayEquals(f, l, 0);
		});

		assertPermutationDR(df, result.pFeatures);
		assertPermutationDR(df, result.pLabels);
	}

	private void assertPermutationDR(double[] df, List<MatrixObject> list) {
		Map<Double, Integer> dict = new HashMap<>();
		for (double d : df) {
			dict.put(d, 0);
		}
		IntStream.range(0, list.size()).forEach(i -> {
			double[] f = list.get(i).acquireRead().getDenseBlockValues();
			for (double d : f) {
				dict.compute(d, (k, v) -> v + 1);
			}
		});

		// check if all the occurrences are equivalent to one
		for (Map.Entry<Double, Integer> e : dict.entrySet()) {
			Assert.assertEquals(1, (int) e.getValue());
		}
	}

	@Test
	public void testDataPartitionerDRR() {
		DataPartitionerLocal dp = new DataPartitionerLocal(Statement.PSScheme.DISJOINT_ROUND_ROBIN);
		double[] df = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		DataPartitionLocalScheme.Result result = dp.doPartitioning(4, DataConverter.convertToMatrixBlock(df, true),
			DataConverter.convertToMatrixBlock(df, true));

		Assert.assertEquals(4, result.pFeatures.size());
		Assert.assertEquals(4, result.pLabels.size());

		double[] expected1 = new double[] { 4, 8 };
		assertResult(result, 0, expected1);

		double[] expected2 = new double[] { 1, 5, 9 };
		assertResult(result, 1, expected2);

		double[] expected3 = new double[] { 2, 6, 10 };
		assertResult(result, 2, expected3);

		double[] expected4 = new double[] { 3, 7 };
		assertResult(result, 3, expected4);
	}

	@Test
	public void testDataPartitionerOR() {
		DataPartitionerLocal dp = new DataPartitionerLocal(Statement.PSScheme.OVERLAP_RESHUFFLE);
		double[] df = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		DataPartitionLocalScheme.Result result = dp.doPartitioning(4, DataConverter.convertToMatrixBlock(df, true),
			DataConverter.convertToMatrixBlock(df, true));

		Assert.assertEquals(4, result.pFeatures.size());
		Assert.assertEquals(4, result.pLabels.size());

		assertPermutationOR(df, result.pFeatures);
		assertPermutationOR(df, result.pLabels);
	}

	private void assertPermutationOR(double[] df, List<MatrixObject> list) {
		for (MatrixObject mo : list) {
			Map<Double, Integer> dict = new HashMap<>();
			for (double d : df) {
				dict.put(d, 0);
			}
			double[] f = mo.acquireRead().getDenseBlockValues();
			for (double d : f) {
				dict.compute(d, (k, v) -> v + 1);
			}
			Assert.assertEquals(10, dict.size());
			// check if all the occurrences are equivalent to one
			for (Map.Entry<Double, Integer> e : dict.entrySet()) {
				Assert.assertEquals(1, (int) e.getValue());
			}
		}
	}
}

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

package org.apache.sysds.test.component.compress.qcompress;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.apache.commons.lang3.tuple.Pair;

/**
 * This class tests the quantization-fused compression in SystemDS.
 */
public class QuantizationFusedCompressionTest {

	/**
	 * Test 1: Quantization-fused Compression with a scalar scaling factor.
	 */
	@Test
	public void testQuantizationCompressionWithScalar() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(4, 4, 1, 10, 1.0, 1234);
		ScalarObject sf = new DoubleObject(2.5);
		Pair<MatrixBlock, CompressionStatistics> result = CompressedMatrixBlockFactory.compress(mb, sf, 1, null);
		MatrixBlock qmb = result.getLeft();
		for(int i = 0; i < mb.getNumRows(); i++) {
			for(int j = 0; j < mb.getNumColumns(); j++) {
				double expected = Math.floor(mb.get(i, j) * sf.getDoubleValue());
				assertEquals("Quantized compression mismatch!", expected, qmb.get(i, j), 0.0);
			}
		}
	}

	/**
	 * Test 2: Quantization-fused compression with row-wise vector scaling.
	 */
	@Test
	public void testQuantizationCompressionWithRowwiseVectorScale() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(5, 4, 1, 10, 1.0, 5678);
		MatrixBlock sf = new MatrixBlock(5, 1, false);
		sf.set(0, 0, 1.5);
		sf.set(1, 0, 2.0);
		sf.set(2, 0, 2.5);
		sf.set(3, 0, 3.0);
		sf.set(4, 0, 3.5);
		Pair<MatrixBlock, CompressionStatistics> result = CompressedMatrixBlockFactory.compress(mb, sf, 1, null);
		MatrixBlock qmb = result.getLeft();
		for(int i = 0; i < mb.getNumRows(); i++) {
			for(int j = 0; j < mb.getNumColumns(); j++) {
				double expected = Math.floor(mb.get(i, j) * sf.get(i, 0));
				assertEquals("Quantized compression mismatch!", expected, qmb.get(i, j), 0.0);
			}
		}
	}

	/**
	 * Test 3: Compare compression statistics of two matrices, m0 and m1, where m0 is derived as m0 = floor(m1 * sf) with
	 * sf = 0.5.
	 * 
	 * - Compression for m0 is aborted at phase 1 (before co-code). - Compression for m1 should also be aborted at the
	 * same phase. - The resulting compression statistics for both matrices should match.
	 */
	@Test
	public void testQuantizationFusedCompressionAbortedBeforeCoCodeStats() {
		double[][] values0 = {{0, 1, 1, 2, 2}, {3, 3, 4, 4, 5}, {5, 6, 6, 7, 7}, {8, 8, 9, 9, 10}, {10, 11, 11, 12, 12},
			{13, 13, 14, 14, 15}};
		MatrixBlock m0 = DataConverter.convertToMatrixBlock(values0);
		m0.recomputeNonZeros();

		Pair<MatrixBlock, CompressionStatistics> cm0 = CompressedMatrixBlockFactory.compress(m0);
		CompressionStatistics stats0 = cm0.getRight();

		MatrixBlock m1 = new MatrixBlock(6, 5, false);
		int val = 1;
		for(int i = 0; i < 6; i++) {
			for(int j = 0; j < 5; j++) {
				m1.set(i, j, val++);
			}
		}
		m1.recomputeNonZeros();

		DoubleObject sf = new DoubleObject(0.5);
		Pair<MatrixBlock, CompressionStatistics> cm1 = CompressedMatrixBlockFactory.compress(m1, sf, 1, null);
		CompressionStatistics stats1 = cm1.getRight();

		assertTrue("Compression statistics must match", stats0.toString().equals(stats1.toString()));
		// Since m0 and m1 have different values their number of non-zero values is different
		// assertEquals("Non-zero count should match", m0.getNonZeros(), m1.getNonZeros(), 0.1);
	}

	/**
	 * Test 4: Compare compression statistics of two matrices, m0 and m1, where m0 is derived as m0 = floor(m1 * sf) with
	 * sf = 0.3.
	 * 
	 * - Compression for m0 is aborted at phase 2 (after co-code). - Compression for m1 should also be aborted at the
	 * same phase. - The resulting compression statistics for both matrices should match.
	 */
	@Test
	public void testQuantizationFusedCompressionAbortedAfterCoCodeStats() {
		double[][] values1 = {{1, 8, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6},
			{3, 4, 5, 6, 7}};
		MatrixBlock m1 = DataConverter.convertToMatrixBlock(values1);
		m1.recomputeNonZeros();

		double scaleFactor = 0.3;
		MatrixBlock m0 = new MatrixBlock(m1.getNumRows(), m1.getNumColumns(), false);
		for(int i = 0; i < m1.getNumRows(); i++) {
			for(int j = 0; j < m1.getNumColumns(); j++) {
				m0.set(i, j, Math.floor(m1.get(i, j) * scaleFactor));
			}
		}
		m0.recomputeNonZeros();

		Pair<MatrixBlock, CompressionStatistics> cm0 = CompressedMatrixBlockFactory.compress(m0);
		CompressionStatistics stats0 = cm0.getRight();

		DoubleObject sf = new DoubleObject(scaleFactor);
		Pair<MatrixBlock, CompressionStatistics> cm1 = CompressedMatrixBlockFactory.compress(m1, sf, 1, null);
		CompressionStatistics stats1 = cm1.getRight();

		assertTrue("Compression statistics must match", stats0.toString().equals(stats1.toString()));
		// Since m0 and m1 have different values their number of non-zero values is different
		// assertEquals("Non-zero count should match", m0.getNonZeros(), m1.getNonZeros(), 0.1);
	}
}

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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;
import static org.mockito.ArgumentMatchers.refEq;

import java.util.List;
import java.util.Random;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.cocode.CoCoderFactory.PartitionerType;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.estim.AComEst;
import org.apache.sysds.runtime.compress.estim.ComEstFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class CompareCompressionTypeTest {

	/**
	 * Test 1: Compare the best compression types of two matrices, m0 and m1.
	 * 
	 * - m0 is generated as a floored matrix. 
	 * - m1 is generated as a full-precision matrix, but will be internally multiplied by 1.0 and floored.
	 * - Since m1 undergoes an equivalent transformation (scaling by 1.0 and flooring), the best compression types 
	 *   determined by the estimator should match elementwise for both matrices.
	 * - This validates that the estimator correctly handles explicit flooring vs. internal scaling and flooring during quantization-fused compression.
	 */
	@Test
	public void testCompareBestCompressionTypeForTwoMatrices() {
		try {
			Random r = new Random(1234);
			int k = 4;

			// Generate first floored matrix and compute compression info
			MatrixBlock m0 = generateTestMatrix(10000, 500, 1, 100, 1.0, r, true);
			CompressionSettings cs0 = new CompressionSettingsBuilder().setColumnPartitioner(PartitionerType.GREEDY)
				.setSeed(1234).create();
			AComEst estimator0 = ComEstFactory.createEstimator(m0, cs0, k);
			CompressedSizeInfo compressedGroups0 = estimator0.computeCompressedSizeInfos(k);

			// Generate second matrix full-precision matrix that will be internally scaled by 1.0 and floored and compute
			// compression info
			MatrixBlock m1 = generateTestMatrix(10000, 500, 1, 100, 1.0, r, false);
			double[] scaleFactor = {1.0};
			CompressionSettings cs1 = new CompressionSettingsBuilder().setColumnPartitioner(PartitionerType.GREEDY)
				.setScaleFactor(scaleFactor).setSeed(1234).create();
			AComEst estimator1 = ComEstFactory.createEstimator(m1, cs1, k);
			CompressedSizeInfo compressedGroups1 = estimator1.computeCompressedSizeInfos(k);

			List<CompressedSizeInfoColGroup> groups0 = compressedGroups0.getInfo();
			List<CompressedSizeInfoColGroup> groups1 = compressedGroups1.getInfo();

			assertEquals("Mismatch in number of compressed groups", groups0.size(), groups1.size());

			for(int i = 0; i < groups0.size(); i++) {
				assertEquals("Best compression type mismatch at index " + i, groups0.get(i).getBestCompressionType(), groups1.get(i).getBestCompressionType());
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Compression extraction failed: " + e.getMessage());
		}
	}

	/**
	 * Generate a test matrix with specified dimensions, value range, and sparsity.
	 */
	private static MatrixBlock generateTestMatrix(int nRow, int nCol, int min, int max, double s, Random r,
		boolean floored) {
		final int m = Integer.MAX_VALUE;
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(nRow, nCol, min, max, s, r.nextInt(m));
		return floored ? TestUtils.floor(mb) : mb;
	}

}

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

package org.apache.sysds.test.component.compress;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.Random;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.lib.CLALibSort;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * Tests the single-column sort of compressed column groups via {@link CLALibSort}. The compressed result is expected to
 * match an ascending sort of the original column values for every implemented column-group encoding.
 */
public class CompressedSortTest {

	private static final int ROWS = 1000;

	@Test
	public void sortDDC() {
		// dense, few unique, positive -> DDC
		run(generate(ROWS, 8, 1.0, -1, 50, 1, 7), CompressionType.DDC);
	}

	@Test
	public void sortDDCWithNegatives() {
		// dense, few unique, spanning negative/positive -> DDC
		run(generate(ROWS, 10, 1.0, -25, 25, 0, 13), CompressionType.DDC);
	}

	@Test
	public void sortSDCZeros() {
		// sparse, few unique, positive -> SDC variant with zero default
		run(generate(ROWS, 6, 0.2, 1, 40, 0, 23), CompressionType.SDC);
	}

	@Test
	public void sortSDCWithNegatives() {
		// sparse, few unique, spanning negative/zero/positive -> SDC variant
		run(generate(ROWS, 8, 0.3, -20, 20, 0, 41), CompressionType.SDC);
	}

	@Test
	public void sortSDCSingleValue() {
		// sparse with a single distinct non-zero value -> SDCSingleZeros
		run(generate(ROWS, 1, 0.25, 5, 5, 0, 99), CompressionType.SDC);
	}

	@Test
	public void sortSDCSingleNonZeroDefault() {
		// two distinct non-zero values, one dominant default -> SDCSingle
		MatrixBlock mb = new MatrixBlock(ROWS, 1, false);
		for(int i = 0; i < ROWS; i++)
			mb.set(i, 0, i % 10 < 3 ? 3 : 7);
		mb.recomputeNonZeros();
		run(mb, CompressionType.SDC);
	}

	@Test
	public void sortSDCSingleNonZeroDefaultNegative() {
		// dominant non-zero default with a single smaller (negative) value -> SDCSingle
		MatrixBlock mb = new MatrixBlock(ROWS, 1, false);
		for(int i = 0; i < ROWS; i++)
			mb.set(i, 0, i % 10 < 3 ? -4 : 7);
		mb.recomputeNonZeros();
		run(mb, CompressionType.SDC);
	}

	@Test
	public void sortConst() {
		// constant column -> CONST, sort is a no-op
		MatrixBlock mb = new MatrixBlock(ROWS, 1, false);
		for(int i = 0; i < ROWS; i++)
			mb.set(i, 0, 3);
		mb.recomputeNonZeros();
		run(mb, CompressionType.CONST);
	}

	private void run(MatrixBlock mb, CompressionType ct) {
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0)
			.setValidCompressions(EnumSet.of(ct));
		MatrixBlock compressed = CompressedMatrixBlockFactory.compress(mb, 1, csb).getLeft();
		assertTrue("Expected the input to compress into a " + ct + " backed block",
			compressed instanceof CompressedMatrixBlock);
		CompressedMatrixBlock cmb = (CompressedMatrixBlock) compressed;
		assertEquals("Expected a single column group", 1, cmb.getColGroups().size());

		MatrixBlock actual = CompressedMatrixBlock.getUncompressed(CLALibSort.sort(cmb, null, null, 1), "sort");
		MatrixBlock expected = referenceSort(mb);

		TestUtils.compareMatrices(expected, actual, 0.0, "sort " + ct);
	}

	private static MatrixBlock referenceSort(MatrixBlock mb) {
		final int n = mb.getNumRows();
		double[] v = new double[n];
		for(int i = 0; i < n; i++)
			v[i] = mb.get(i, 0);
		Arrays.sort(v);
		MatrixBlock e = new MatrixBlock(n, 1, false);
		for(int i = 0; i < n; i++)
			e.set(i, 0, v[i]);
		e.recomputeNonZeros();
		return e;
	}

	private static MatrixBlock generate(int rows, int unique, double sparsity, int min, int max, int seed,
		int valueSeed) {
		final MatrixBlock mb = new MatrixBlock(rows, 1, false);
		final Random pos = new Random(seed);
		final Random val = new Random(valueSeed);
		final double[] values = new double[Math.max(unique, 1)];
		for(int i = 0; i < values.length; i++)
			values[i] = min + (max > min ? val.nextInt(max - min + 1) : 0);
		for(int i = 0; i < rows; i++)
			if(pos.nextDouble() < sparsity)
				mb.set(i, 0, values[pos.nextInt(values.length)]);
		mb.recomputeNonZeros();
		return mb;
	}
}

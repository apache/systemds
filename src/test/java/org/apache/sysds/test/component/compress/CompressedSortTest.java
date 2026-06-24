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

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.functionobjects.SortIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * Tests the {@code order} (sort) reorg operation on compressed matrices. A single column held in a single column group
 * is sorted ascending while staying compressed (via {@link org.apache.sysds.runtime.compress.lib.CLALibSort}); every
 * other configuration falls back to a decompressed reorg. In all cases the result must match the uncompressed reference.
 */
public class CompressedSortTest {

	private static final int ROWS = 1000;

	private static final ReorgOperator ASC = new ReorgOperator(new SortIndex(1, false, false), 1);
	private static final ReorgOperator DESC = new ReorgOperator(new SortIndex(1, true, false), 1);

	@Test
	public void sortDDC() {
		runCompressed(generate(ROWS, 1, 8, 1.0, 1, 50, 7), CompressionType.DDC);
	}

	@Test
	public void sortDDCWithNegatives() {
		runCompressed(generate(ROWS, 1, 10, 1.0, -25, 25, 13), CompressionType.DDC);
	}

	@Test
	public void sortSDCZeros() {
		runCompressed(generate(ROWS, 1, 6, 0.2, 1, 40, 23), CompressionType.SDC);
	}

	@Test
	public void sortSDCWithNegatives() {
		runCompressed(generate(ROWS, 1, 8, 0.3, -20, 20, 41), CompressionType.SDC);
	}

	@Test
	public void sortSDCSingleValueZeros() {
		// sparse with a single distinct non-zero value -> SDCSingleZeros
		runCompressed(generate(ROWS, 1, 1, 0.25, 5, 5, 99), CompressionType.SDC);
	}

	@Test
	public void sortSDCSingleNonZeroDefault() {
		// two distinct non-zero values, one dominant default -> SDCSingle
		runCompressed(twoValueColumn(3, 7), CompressionType.SDC);
	}

	@Test
	public void sortSDCSingleNonZeroDefaultNegative() {
		// dominant non-zero default with a single smaller (negative) value -> SDCSingle
		runCompressed(twoValueColumn(-4, 7), CompressionType.SDC);
	}

	@Test
	public void sortConst() {
		MatrixBlock mb = new MatrixBlock(ROWS, 1, false);
		for(int i = 0; i < ROWS; i++)
			mb.set(i, 0, 3);
		mb.recomputeNonZeros();
		runCompressed(mb, CompressionType.CONST);
	}

	@Test
	public void sortUncompressedColGroup() {
		// a CompressedMatrixBlock holding a single uncompressed column group must also sort correctly
		MatrixBlock raw = generate(ROWS, 1, ROWS, 1.0, -100000, 100000, 5);
		List<AColGroup> groups = new ArrayList<>(1);
		groups.add(ColGroupUncompressed.create(raw, ColIndexFactory.create(1)));
		CompressedMatrixBlock cmb = new CompressedMatrixBlock(ROWS, 1, raw.getNonZeros(), false, groups);

		MatrixBlock actual = cmb.reorgOperations(ASC, new MatrixBlock(), 0, 0, 0);
		assertTrue("Expected the sorted result to stay compressed", actual instanceof CompressedMatrixBlock);
		MatrixBlock expected = raw.reorgOperations(ASC, new MatrixBlock(), 0, 0, 0);
		TestUtils.compareMatrices(expected, CompressedMatrixBlock.getUncompressed(actual, "sort"), 0.0,
			"sort UNCOMPRESSED colgroup");
	}

	@Test
	public void sortDescendingFallback() {
		// descending order is not supported by the compressed fast-path -> decompress fallback
		runFallback(generate(ROWS, 1, 8, 1.0, 1, 50, 7), CompressionType.DDC, DESC);
	}

	@Test
	public void sortMultiColumnFallback() {
		// order on a multi-column matrix sorts rows by the first column -> decompress fallback
		runFallback(generate(ROWS, 3, 6, 1.0, 1, 30, 31), CompressionType.DDC, ASC);
	}

	@Test
	public void quantileTableDDC() {
		runQuantile(generate(ROWS, 1, 8, 1.0, 1, 50, 7), CompressionType.DDC);
	}

	@Test
	public void quantileTableDDCWithNegatives() {
		runQuantile(generate(ROWS, 1, 10, 1.0, -25, 25, 13), CompressionType.DDC);
	}

	@Test
	public void quantileTableSDCZeros() {
		runQuantile(generate(ROWS, 1, 6, 0.2, 1, 40, 23), CompressionType.SDC);
	}

	@Test
	public void quantileTableSDCWithNegatives() {
		runQuantile(generate(ROWS, 1, 8, 0.3, -20, 20, 41), CompressionType.SDC);
	}

	@Test
	public void quantileTableAllNegative() {
		runQuantile(generate(ROWS, 1, 8, 0.4, -50, -1, 57), CompressionType.SDC);
	}

	@Test
	public void quantileTableConst() {
		MatrixBlock mb = new MatrixBlock(ROWS, 1, false);
		for(int i = 0; i < ROWS; i++)
			mb.set(i, 0, 3);
		mb.recomputeNonZeros();
		runQuantile(mb, CompressionType.CONST);
	}

	@Test
	public void quantileWeightedFallback() {
		MatrixBlock mb = generate(ROWS, 1, 8, 1.0, 1, 50, 7);
		MatrixBlock weights = new MatrixBlock(ROWS, 1, false);
		Random r = new Random(123);
		for(int i = 0; i < ROWS; i++)
			weights.set(i, 0, r.nextInt(4) + 1);
		weights.recomputeNonZeros();
		MatrixBlock expected = new MatrixBlock(mb).sortOperations(weights, new MatrixBlock(), 1);

		CompressedMatrixBlock cmb = compress(mb, CompressionType.DDC);
		MatrixBlock actual = cmb.sortOperations(weights, new MatrixBlock(), 1);

		expected.recomputeNonZeros();
		actual.recomputeNonZeros();
		TestUtils.compareMatrices(expected, actual, 0.0, "weighted sortOperations fallback");
	}

	private void runQuantile(MatrixBlock mb, CompressionType ct) {
		// reference is computed on a copy because compression may consume the input.
		MatrixBlock expected = new MatrixBlock(mb).sortOperations(null, new MatrixBlock(), 1);

		CompressedMatrixBlock cmb = compress(mb, ct);
		assertEquals("Expected a single column group", 1, cmb.getColGroups().size());

		MatrixBlock actual = cmb.sortOperations(null, new MatrixBlock(), 1);

		// sortOperations leaves the non-zero count unmaintained; recompute so the value comparison reads the data.
		expected.recomputeNonZeros();
		actual.recomputeNonZeros();

		// the value/weight table must match the uncompressed reference bit-for-bit ...
		TestUtils.compareMatrices(expected, actual, 0.0, "sortOperations table " + ct);
		// ... so the downstream median/quantile picks are identical.
		assertEquals("median " + ct, expected.median(), actual.median(), 0.0);
		assertEquals("q25 " + ct, expected.pickValue(0.25), actual.pickValue(0.25), 0.0);
		assertEquals("q90 " + ct, expected.pickValue(0.90), actual.pickValue(0.90), 0.0);
	}

	private void runCompressed(MatrixBlock mb, CompressionType ct) {
		CompressedMatrixBlock cmb = compress(mb, ct);
		assertEquals("Expected a single column group", 1, cmb.getColGroups().size());

		MatrixBlock actual = cmb.reorgOperations(ASC, new MatrixBlock(), 0, 0, 0);
		assertTrue("Expected the sorted result to stay compressed for " + ct,
			actual instanceof CompressedMatrixBlock);

		MatrixBlock expected = mb.reorgOperations(ASC, new MatrixBlock(), 0, 0, 0);
		TestUtils.compareMatrices(expected, CompressedMatrixBlock.getUncompressed(actual, "sort"), 0.0, "sort " + ct);
	}

	private void runFallback(MatrixBlock mb, CompressionType ct, ReorgOperator op) {
		CompressedMatrixBlock cmb = compress(mb, ct);

		MatrixBlock actual = cmb.reorgOperations(op, new MatrixBlock(), 0, 0, 0);
		MatrixBlock expected = mb.reorgOperations(op, new MatrixBlock(), 0, 0, 0);
		TestUtils.compareMatrices(expected, CompressedMatrixBlock.getUncompressed(actual, "sort"), 0.0,
			"sort fallback " + ct);
	}

	private static CompressedMatrixBlock compress(MatrixBlock mb, CompressionType ct) {
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0)
			.setValidCompressions(EnumSet.of(ct));
		MatrixBlock compressed = CompressedMatrixBlockFactory.compress(mb, 1, csb).getLeft();
		assertTrue("Expected the input to compress into a " + ct + " backed block",
			compressed instanceof CompressedMatrixBlock);
		return (CompressedMatrixBlock) compressed;
	}

	private static MatrixBlock twoValueColumn(int rare, int dominant) {
		MatrixBlock mb = new MatrixBlock(ROWS, 1, false);
		for(int i = 0; i < ROWS; i++)
			mb.set(i, 0, i % 10 < 3 ? rare : dominant);
		mb.recomputeNonZeros();
		return mb;
	}

	private static MatrixBlock generate(int rows, int cols, int unique, double sparsity, int min, int max, int seed) {
		final MatrixBlock mb = new MatrixBlock(rows, cols, false);
		final Random pos = new Random(seed);
		final Random val = new Random(seed * 31 + 1);
		final double[] values = new double[Math.max(unique, 1)];
		for(int i = 0; i < values.length; i++)
			values[i] = min + (max > min ? val.nextInt(max - min + 1) : 0);
		for(int i = 0; i < rows; i++)
			for(int j = 0; j < cols; j++)
				if(pos.nextDouble() < sparsity)
					mb.set(i, j, values[pos.nextInt(values.length)]);
		mb.recomputeNonZeros();
		return mb;
	}
}

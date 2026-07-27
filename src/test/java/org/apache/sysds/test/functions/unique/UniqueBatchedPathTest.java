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

package org.apache.sysds.test.functions.unique;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.LibMatrixSketch;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

/**
 * Covers the batched row-wise and column-wise unique paths, which the script-level tests cannot reach.
 *
 * <p>
 * The row-wise and column-wise workers charge only one live set per thread, so batching is selected only when
 * {@code numThreads * cellsPerIndex * 64} exceeds the budget. Reaching that from an end-to-end run would require the
 * budget to fall into a narrow window that depends on the number of threads of the executing machine, which is not
 * deterministic across CI runners. The batched RowCol path does not have this problem and is covered end-to-end by
 * {@link UniqueRowCol}.
 */
public class UniqueBatchedPathTest {

	@Test
	public void testBatchedRowMatchesBaseline() {
		MatrixBlock in = new MatrixBlock(400, 64, false).allocateBlock();
		for(int i = 0; i < in.getNumRows(); i++)
			for(int j = 0; j < in.getNumColumns(); j++)
				in.set(i, j, (i + j) % 8);
		in.recomputeNonZeros();

		// the full parallel path needs numThreads * clen * 64 bytes, so a budget below that batches
		assertEquivalent(in, Types.Direction.Row, 4, 4 * 64 * 64 / 2);
	}

	@Test
	public void testBatchedColMatchesBaseline() {
		MatrixBlock in = new MatrixBlock(64, 400, false).allocateBlock();
		for(int j = 0; j < in.getNumColumns(); j++)
			for(int i = 0; i < in.getNumRows(); i++)
				in.set(i, j, (i + j) % 8);
		in.recomputeNonZeros();

		assertEquivalent(in, Types.Direction.Col, 4, 4 * 64 * 64 / 2);
	}

	/**
	 * Checks that the batched result is identical to the single-threaded baseline.
	 */
	private static void assertEquivalent(MatrixBlock in, Types.Direction dir, int k, long maxLocalBytes) {
		MatrixBlock expected = LibMatrixSketch.getUniqueValues(in, dir);
		MatrixBlock actual = LibMatrixSketch.getUniqueValues(in, dir, k, maxLocalBytes);

		assertEquals("number of rows", expected.getNumRows(), actual.getNumRows());
		assertEquals("number of columns", expected.getNumColumns(), actual.getNumColumns());
		for(int i = 0; i < expected.getNumRows(); i++)
			for(int j = 0; j < expected.getNumColumns(); j++)
				assertEquals("value at (" + i + ", " + j + ")", expected.get(i, j), actual.get(i, j), 0);
	}
}

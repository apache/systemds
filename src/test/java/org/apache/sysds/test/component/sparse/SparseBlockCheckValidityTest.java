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

package org.apache.sysds.test.component.sparse;

import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCOO;
import org.apache.sysds.runtime.data.SparseBlockCSC;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockDCSR;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.data.SparseBlockMCSC;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.data.SparseRowVector;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.lang.reflect.Field;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class SparseBlockCheckValidityTest extends AutomatedTestBase
{
	private final static int _rows = 123;
	private final static int _cols = 97;
	private final static double _sparsity = 0.22;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockCOOValid() {
		runSparseBlockValidTest(SparseBlock.Type.COO);
	}

	@Test
	public void testSparseBlockCSCValid() {
		runSparseBlockValidTest(SparseBlock.Type.CSC);
	}

	@Test
	public void testSparseBlockCSRValid() {
		runSparseBlockValidTest(SparseBlock.Type.CSR);
	}

	@Test
	public void testSparseBlockDCSRValid() {
		runSparseBlockValidTest(SparseBlock.Type.DCSR);
	}

	@Test
	public void testSparseBlockMCSCValid() {
		runSparseBlockValidTest(SparseBlock.Type.MCSC);
	}

	@Test
	public void testSparseBlockMCSRValid() {
		runSparseBlockValidTest(SparseBlock.Type.MCSR);
	}

	@Test
	public void testSparseBlockCOOInvalidDimensions() {
		runSparseBlockInvalidDimensionsTest(new SparseBlockCOO(-1, 0));
	}

	@Test
	public void testSparseBlockCSCInvalidDimensions() {
		runSparseBlockInvalidDimensionsTest(new SparseBlockCSC(-1, 0));
	}

	@Test
	public void testSparseBlockCSRInvalidDimensions() {
		runSparseBlockInvalidDimensionsTest(new SparseBlockCSR(-1, 0));
	}

	@Test
	public void testSparseBlockDCSRInvalidDimensions() {
		runSparseBlockInvalidDimensionsTest(new SparseBlockDCSR(0, 0));
	}

	@Test
	public void testSparseBlockMCSCInvalidDimensions() {
		runSparseBlockInvalidDimensionsTest(new SparseBlockMCSC(-1, 0));
	}

	@Test
	public void testSparseBlockMCSRInvalidDimensions() {
		runSparseBlockInvalidDimensionsTest(new SparseBlockMCSR(0, -1));
	}

	@Test
	public void testSparseBlockCOOIncorrectArrayLengths() {
		SparseBlockCOO sblock = new SparseBlockCOO(2, 2);
		// nnz > capacity
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(2, 2, 4, false));

		assertEquals("Incorrect array lengths.", ex.getMessage());
	}

	@Test
	public void testSparseBlockCSCIncorrectArrayLengths() {
		SparseBlockCSC sblock = new SparseBlockCSC(2, 2, 2);
		// nnz > capacity
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(2, 3, 6, false));

		assertEquals("Incorrect array lengths.", ex.getMessage());
	}

	@Test
	public void testSparseBlockCSRIncorrectArrayLengths() {
		SparseBlockCSR sblock = new SparseBlockCSR(2, 2, 1);
		// nnz > capacity
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(3, 2, 6, false));

		assertEquals("Incorrect array lengths.", ex.getMessage());
	}

	@Test
	public void testSparseBlockDCSRIncorrectArrayLengths() {
		SparseBlockDCSR sblock = new SparseBlockDCSR(2, 1);

		// cut off last value
		int[] rowptr = (int[]) getField(sblock,"_rowptr");
		setField(sblock, "_rowptr", Arrays.copyOfRange(rowptr, 0, rowptr.length-1));
		// nnz > capacity
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(3, 2, 6, false));

		assertEquals("Incorrect array lengths.", ex.getMessage());
	}

	@Test
	public void testSparseBlockMCSCIncorrectArrayLengths() {
		SparseBlockMCSC sblock = new SparseBlockMCSC(2, 2);

		// nnz > capacity
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(3, 2, 1, false));

		assertTrue(ex.getMessage().startsWith("Incorrect size"));
	}

	@Test
	public void testSparseBlockMCSRIncorrectArrayLengths() {
		SparseBlockMCSR sblock = new SparseBlockMCSR(2, 2);

		// nnz > capacity
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(3, 2, 1, false));

		assertTrue(ex.getMessage().startsWith("Incorrect size"));
	}

	@Test
	public void testSparseBlockCOOUnsortedRowIndices() {
		SparseBlockCOO block = new SparseBlockCOO(10, 3);

		int[] r = new int[]{0, 5, 2}; // unsorted
		int[] c = new int[]{0, 1, 2};
		double[] v = new double[]{1, 1, 1};

		setField(block, "_rindexes", r);
		setField(block, "_cindexes", c);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(10, 10, 3, false));

		assertEquals("Wrong sorted order of row indices", ex.getMessage());
	}

	@Test
	public void testSparseBlockCSCDecreasingColPointers() {
		SparseBlockCSC block = new SparseBlockCSC(10, 3);

		int[] ptr = new int[]{0, 2, 1, 3}; // unsorted col pointer
		int[] idxs = new int[]{0, 1, 2};
		double[] v = new double[]{1, 1, 1};

		setField(block, "_ptr", ptr);
		setField(block, "_indexes", idxs);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(10, 3, 3, true));

		assertTrue(ex.getMessage().startsWith("Column pointers are decreasing at column"));
	}

	@Test
	public void testSparseBlockCSRDecreasingRowPointers() {
		SparseBlockCSR block = new SparseBlockCSR(3, 3);

		int[] ptr = new int[]{0, 2, 1, 3}; // unsorted row pointer
		int[] idxs = new int[]{0, 1, 2};
		double[] v = new double[]{1, 1, 1};

		setField(block, "_ptr", ptr);
		setField(block, "_indexes", idxs);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 10, 3, true));

		assertTrue(ex.getMessage().startsWith("Row pointers are decreasing at row"));
	}

	@Test
	public void testSparseBlockDCSRDecreasingRowIndices() {
		SparseBlockDCSR block = new SparseBlockDCSR(3, 3);

		int[] rowIdxs = new int[]{0, 2, 1}; // unsorted
		int[] rowPtr = new int[]{0, 1, 2, 3};
		int[] colIdxs = new int[]{0, 1, 2};
		double[] v = new double[]{1, 1, 1};

		setField(block, "_rowidx", rowIdxs);
		setField(block, "_rowptr", rowPtr);
		setField(block, "_colidx", colIdxs);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 10, 3, true));

		assertTrue(ex.getMessage().startsWith("Row indices are decreasing at row"));
	}

	@Test
	public void testSparseBlockDCSRDecreasingRowPointers() {
		SparseBlockDCSR block = new SparseBlockDCSR(3, 3);

		int[] rowIdxs = new int[]{0, 1, 2};
		int[] rowPtr = new int[]{0, 1, 3, 2}; // unsorted
		int[] colIdxs = new int[]{0, 1, 2};
		double[] v = new double[]{1, 1, 1};

		setField(block, "_rowidx", rowIdxs);
		setField(block, "_rowptr", rowPtr);
		setField(block, "_colidx", colIdxs);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 10, 3, true));

		assertTrue(ex.getMessage().startsWith("Row pointers are decreasing at row"));
	}

	@Test
	public void testSparseBlockCOOUnsortedColumnIndicesWithinRow() {
		SparseBlockCOO block = new SparseBlockCOO(1, 3);

		int[] r = new int[]{0, 0, 0};
		int[] c = new int[]{0, 2, 1}; // unsorted for row 0
		double[] v = new double[]{1, 1, 1};

		setField(block, "_rindexes", r);
		setField(block, "_cindexes", c);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(1, 3, 3, false));

		assertTrue(ex.getMessage().startsWith("Wrong sparse row ordering"));
	}

	@Test
	public void testSparseBlockCSCUnsortedRowIndicesWithinColumn() {
		SparseBlockCSC block = new SparseBlockCSC(10, 3);

		int[] ptr = new int[]{0, 3, 3, 3};
		int[] idxs = new int[]{0, 2, 1}; // unsorted
		double[] v = new double[]{1, 1, 1};

		setField(block, "_ptr", ptr);
		setField(block, "_indexes", idxs);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(10, 3, 3, true));

		assertTrue(ex.getMessage().startsWith("Wrong sparse column ordering"));
	}

	@Test
	public void testSparseBlockCSRUnsortedColumnIndicesWithinRow() {
		SparseBlockCSR block = new SparseBlockCSR(3, 3);

		int[] ptr = new int[]{0, 3, 3, 3};
		int[] idxs = new int[]{0, 2, 1}; // unsorted
		double[] v = new double[]{1, 1, 1};

		setField(block, "_ptr", ptr);
		setField(block, "_indexes", idxs);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(1, 3, 3, false));

		assertTrue(ex.getMessage().startsWith("Wrong sparse row ordering"));
	}

	@Test
	public void testSparseBlockDCSRUnsortedColumnIndicesWithinRow() {
		SparseBlockDCSR block = new SparseBlockDCSR(3, 3);

		int[] rowIdxs = new int[]{0, 2};
		int[] rowPtr = new int[]{0, 1, 3};
		int[] colIdxs = new int[]{0, 2, 1}; // for row 2 unsorted
		double[] v = new double[]{1, 1, 1};

		setField(block, "_rowidx", rowIdxs);
		setField(block, "_rowptr", rowPtr);
		setField(block, "_colidx", colIdxs);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(1, 3, 3, false));

		assertTrue(ex.getMessage().startsWith("Wrong sparse row ordering"));
	}

	@Test
	public void testSparseBlockMCSCUnsortedRowIndicesWithinColumn() {
		SparseBlockMCSC block = new SparseBlockMCSC(10, 3);

		SparseRow col = new SparseRowVector(new double[]{1., 1., 1.}, new int[]{0, 2, 1}); // unsorted
		SparseRow[] cols = new SparseRow[]{null, null, col};
		setField(block, "_columns", cols);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(10, 3, 3, true));

		assertTrue(ex.getMessage().startsWith("Wrong sparse column ordering"));
	}

	@Test
	public void testSparseBlockMCSRUnsortedColumnIndicesWithinRow() {
		SparseBlockMCSR block = new SparseBlockMCSR(3, 10);

		SparseRow row = new SparseRowVector(new double[]{1., 1., 1.}, new int[]{0, 2, 1}); // unsorted
		SparseRow[] rows = new SparseRow[]{null, null, row};
		setField(block, "_rows", rows);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 10, 3, true));

		assertTrue(ex.getMessage().startsWith("Wrong sparse row ordering"));
	}

	@Test
	public void testSparseBlockMCSCInvalidIndices() {
		SparseBlockMCSC block = new SparseBlockMCSC(10, 3);

		SparseRow col = new SparseRowVector(new double[]{1., 1., 1.}, new int[]{-1, 0, 2});
		SparseRow[] cols = new SparseRow[]{null, null, col};
		setField(block, "_columns", cols);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(10, 3, 3, true));

		assertTrue(ex.getMessage().startsWith("Invalid index"));
	}

	@Test
	public void testSparseBlockMCSRInvalidIndices() {
		SparseBlockMCSR block = new SparseBlockMCSR(3, 10);

		SparseRow row = new SparseRowVector(new double[]{1., 1., 1.}, new int[]{-1, 0, 1});
		SparseRow[] rows = new SparseRow[]{null, null, row};
		setField(block, "_rows", rows);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 10, 3, true));

		assertTrue(ex.getMessage().startsWith("Invalid index"));
	}

	@Test
	public void testSparseBlockCOOInvalidValue() {
		SparseBlockCOO block = new SparseBlockCOO(3, 3);

		int[] r = new int[]{0, 1, 2};
		int[] c = new int[]{0, 1, 2};
		double[] v = new double[]{1, 2, 0}; // contains 0

		setField(block, "_rindexes", r);
		setField(block, "_cindexes", c);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 3, 3, false));

		assertTrue(ex.getMessage().startsWith("The values array should not contain zeros"));
	}

	@Test
	public void testSparseBlockCSCInvalidValue() {
		SparseBlockCSC block = new SparseBlockCSC(3, 3);

		int[] ptr = new int[]{0, 3, 3, 3};
		int[] idxs = new int[]{0, 1, 2};
		double[] v = new double[]{1, 1, 0};

		setField(block, "_ptr", ptr);
		setField(block, "_indexes", idxs);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 3, 3, false));

		assertTrue(ex.getMessage().startsWith("The values array should not contain zeros"));
	}

	@Test
	public void testSparseBlockCSRInvalidValue() {
		SparseBlockCSR block = new SparseBlockCSR(3, 3);

		int[] ptr = new int[]{0, 3, 3, 3};
		int[] idxs = new int[]{0, 1, 2};
		double[] v = new double[]{1, 1, 0};

		setField(block, "_ptr", ptr);
		setField(block, "_indexes", idxs);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 3, 3, false));

		assertTrue(ex.getMessage().startsWith("The values array should not contain zeros"));
	}

	@Test
	public void testSparseBlockDCSRInvalidValue() {
		SparseBlockDCSR block = new SparseBlockDCSR(3, 3);

		int[] rowIdxs = new int[]{0, 1, 2};
		int[] rowPtr = new int[]{0, 1, 2, 3};
		int[] colIdxs = new int[]{0, 1, 2};
		double[] v = new double[]{1, 1, 0};

		setField(block, "_rowidx", rowIdxs);
		setField(block, "_rowptr", rowPtr);
		setField(block, "_colidx", colIdxs);
		setField(block, "_values", v);
		setField(block, "_size", 3);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(1, 3, 3, false));

		assertTrue(ex.getMessage().startsWith("The values array should not contain zeros"));
	}

	@Test
	public void testSparseBlockMCSCInvalidValue() {
		SparseBlockMCSC block = new SparseBlockMCSC(10, 3);

		SparseRow col = new SparseRowVector(new double[]{1., 1., 0.}, new int[]{0, 1, 2});
		SparseRow[] cols = new SparseRow[]{null, null, col};
		setField(block, "_columns", cols);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(10, 3, 3, true));

		assertTrue(ex.getMessage().startsWith("The values are expected to be non zeros"));
	}

	@Test
	public void testSparseBlockMCSRInvalidValue() {
		SparseBlockMCSR block = new SparseBlockMCSR(3, 10);

		SparseRow row = new SparseRowVector(new double[]{1., 1., 0.}, new int[]{0, 1, 2});
		SparseRow[] rows = new SparseRow[]{null, null, row};
		setField(block, "_rows", rows);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 10, 3, true));

		assertTrue(ex.getMessage().startsWith("The values are expected to be non zeros"));
	}

	@Test
	public void testSparseBlockCOOCapacityExceedsAllowedLimit() {
		SparseBlockCOO block = new SparseBlockCOO(3, 50);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 3, 0, false));

		// RESIZE_FACTOR1 is 2
		assertTrue(ex.getMessage().startsWith("Capacity is larger than the nnz times a resize factor"));
	}

	@Test
	public void testSparseBlockCSCCapacityExceedsAllowedLimit() {
		SparseBlockCSC block = new SparseBlockCSC(3, 3, 50);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 3, 0, false));

		// RESIZE_FACTOR1 is 2
		assertTrue(ex.getMessage().startsWith("Capacity is larger than the nnz times a resize factor"));
	}

	@Test
	public void testSparseBlockCSRCapacityExceedsAllowedLimit() {
		SparseBlockCSR block = new SparseBlockCSR(3, 50, 0);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 3, 0, false));

		// RESIZE_FACTOR1 is 2
		assertTrue(ex.getMessage().startsWith("Capacity is larger than the nnz times a resize factor"));
	}

	@Test
	public void testSparseBlockDCSRCapacityExceedsAllowedLimit() {
		SparseBlockDCSR block = new SparseBlockDCSR(3, 50);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 3, 0, false));

		// RESIZE_FACTOR1 is 2
		assertTrue(ex.getMessage().startsWith("Capacity is larger than the nnz times a resize factor"));
	}

	@Test
	public void testSparseBlockMCSCCapacityExceedsAllowedLimit() {
		SparseBlockMCSC block = new SparseBlockMCSC(10, 3);

		SparseRow col = new SparseRowVector(new double[]{1., 1., 1., 1., 1.}, new int[]{0, 1, 2, 3, 4});
		SparseRow[] cols = new SparseRow[]{null, null, col};
		setField(block, "_columns", cols);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(10, 3, 2, true));

		assertTrue(ex.getMessage().startsWith("The capacity is larger than nnz times a resize factor"));
	}

	@Test
	public void testSparseBlockMCSRCapacityExceedsAllowedLimit() {
		SparseBlockMCSR block = new SparseBlockMCSR(3, 10);

		SparseRow row = new SparseRowVector(new double[]{1., 1., 1., 1., 1.}, new int[]{0, 1, 2, 3, 4});
		SparseRow[] rows = new SparseRow[]{null, null, row};
		setField(block, "_rows", rows);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> block.checkValidity(3, 10, 2, true));

		assertTrue(ex.getMessage().startsWith("The capacity is larger than nnz times a resize factor"));
	}

	private void runSparseBlockValidTest(SparseBlock.Type btype)  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 13);

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);

		assertTrue("should pass checkValidity", sblock.checkValidity(_rows, _cols, sblock.size(), true));
	}

	private void runSparseBlockInvalidDimensionsTest(SparseBlock block) {
		RuntimeException ex1 = assertThrows(RuntimeException.class,
			() -> block.checkValidity(-1, 1, 0, false));
		assertTrue(ex1.getMessage().startsWith("Invalid block dimensions"));

		RuntimeException ex2 = assertThrows(RuntimeException.class,
			() -> block.checkValidity(1, -1, 0, false));
		assertTrue(ex2.getMessage().startsWith("Invalid block dimensions"));
	}

	private static void setField(Object obj, String name, Object value) {
		try {
			Field f = obj.getClass().getDeclaredField(name);
			f.setAccessible(true);
			f.set(obj, value);
		} catch (Exception ex) {
			throw new RuntimeException("Reflection failed: " + ex.getMessage());
		}
	}

	private static Object getField(Object obj, String name) {
		try {
			Field f = obj.getClass().getDeclaredField(name);
			f.setAccessible(true);
			return f.get(obj);
		} catch (Exception ex) {
			throw new RuntimeException("Reflection failed: " + ex.getMessage());
		}
	}
}

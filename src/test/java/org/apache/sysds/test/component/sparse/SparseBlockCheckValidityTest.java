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

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.lang.reflect.Field;

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
		SparseBlockCOO sblock = new SparseBlockCOO(getFixedSparseBlock());

		int size = (int) sblock.size();
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, size+2, false));
		assertEquals("Incorrect array lengths.", ex.getMessage());

		checkValidityFailsWhenArrayLengthIsTemporarilyModified(sblock, "_cindexes",  new int[size-1]);
		checkValidityFailsWhenArrayLengthIsTemporarilyModified(sblock, "_rindexes",  new int[size-1]);
		checkValidityFailsWhenArrayLengthIsTemporarilyModified(sblock, "_values", new double[size-1]);
	}

	@Test
	public void testSparseBlockCSCIncorrectArrayLengths() {
		SparseBlockCSC sblock = new SparseBlockCSC(getFixedSparseBlock());

		int size = (int) sblock.size();
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, size+2, false));
		assertEquals("Incorrect array lengths.", ex.getMessage());

		int clen = 4;
		checkValidityFailsWhenArrayLengthIsTemporarilyModified(sblock, "_ptr",  new int[clen]); // should be clen+1
		checkValidityFailsWhenArrayLengthIsTemporarilyModified(sblock, "_values", new double[size-1]);
		checkValidityFailsWhenArrayLengthIsTemporarilyModified(sblock, "_indexes",  new int[size-1]);
	}

	@Test
	public void testSparseBlockCSRIncorrectArrayLengths() {
		SparseBlockCSR sblock = new SparseBlockCSR(getFixedSparseBlock());

		int size = (int) sblock.size();
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, size+2, false));
		assertEquals("Incorrect array lengths.", ex.getMessage());

		int rlen = sblock.numRows();
		checkValidityFailsWhenArrayLengthIsTemporarilyModified(sblock, "_ptr",  new int[rlen]); // should be rlen+1
		checkValidityFailsWhenArrayLengthIsTemporarilyModified(sblock, "_values", new double[size-1]);
		checkValidityFailsWhenArrayLengthIsTemporarilyModified(sblock, "_indexes",  new int[size-1]);
	}

	@Test
	public void testSparseBlockDCSRIncorrectArrayLengths() {
		SparseBlockDCSR sblock = new SparseBlockDCSR(getFixedSparseBlock());

		int size = (int) sblock.size();
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, size+2, false));
		assertEquals("Incorrect array lengths.", ex.getMessage());

		int rows = sblock.numRows();
		checkValidityFailsWhenArrayLengthIsTemporarilyModified(sblock, "_rowptr",  new int[rows]); // should be rows+1
		checkValidityFailsWhenArrayLengthIsTemporarilyModified(sblock, "_colidx",  new int[size-1]);
		checkValidityFailsWhenArrayLengthIsTemporarilyModified(sblock, "_values", new double[size-1]);
	}

	@Test
	public void testSparseBlockMCSCIncorrectArrayLengths() {
		SparseBlockMCSC sblock =  new SparseBlockMCSC(getFixedSparseBlock());

		int size = (int) sblock.size();
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, size+2, false));
		assertTrue(ex.getMessage().startsWith("Incorrect size"));
	}

	@Test
	public void testSparseBlockMCSRIncorrectArrayLengths() {
		SparseBlockMCSR sblock = new SparseBlockMCSR(getFixedSparseBlock());

		int size = (int) sblock.size();
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, size+2, false));
		assertTrue(ex.getMessage().startsWith("Incorrect size"));
	}

	@Test
	public void testSparseBlockCOOUnsortedRowIndices() {
		SparseBlockCOO sblock = new SparseBlockCOO(getFixedSparseBlock());
		int[] r = new int[]{0, 2, 1, 2, 3, 3}; // unsorted
		setField(sblock, "_rindexes", r);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, false));
		assertEquals("Wrong sorted order of row indices", ex.getMessage());
	}

	@Test
	public void testSparseBlockCSCDecreasingColPointers() {
		SparseBlockCSC sblock = new SparseBlockCSC(getFixedSparseBlock());
		int[] ptr = new int[]{0, 2, 1, 4, 6}; // unsorted
		setField(sblock, "_ptr", ptr);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, true));
		assertTrue(ex.getMessage().startsWith("Column pointers are decreasing at column"));
	}

	@Test
	public void testSparseBlockCSRDecreasingRowPointers() {
		SparseBlockCSR sblock = new SparseBlockCSR(getFixedSparseBlock());
		int[] ptr = new int[]{0, 2, 1, 4, 6}; // unsorted
		setField(sblock, "_ptr", ptr);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, true));
		assertTrue(ex.getMessage().startsWith("Row pointers are decreasing at row"));
	}

	@Test
	public void testSparseBlockDCSRDecreasingRowIndices() {
		SparseBlockDCSR sblock = new SparseBlockDCSR(getFixedSparseBlock());
		int[] rowIdxs = new int[]{0, 2, 1, 3}; // unsorted
		setField(sblock, "_rowidx", rowIdxs);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, false));
		assertTrue(ex.getMessage().startsWith("Row indices are decreasing at row"));
	}

	@Test
	public void testSparseBlockDCSRDecreasingRowPointers() {
		SparseBlockDCSR sblock = new SparseBlockDCSR(getFixedSparseBlock());
		int[] rowPtr = new int[]{0, 1, 2, 6, 4}; // unsorted
		setField(sblock, "_rowptr", rowPtr);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, false));
		assertTrue(ex.getMessage().startsWith("Row pointers are decreasing at row"));
	}

	@Test
	public void testSparseBlockCOOUnsortedColumnIndicesWithinRow() {
		SparseBlockCOO sblock = new SparseBlockCOO(getFixedSparseBlock());
		int[] c = new int[]{0, 1, 3, 4, 4, 3}; // unsorted for last row
		setField(sblock, "_cindexes", c);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, false));
		assertTrue(ex.getMessage().startsWith("Wrong sparse row ordering"));
	}

	@Test
	public void testSparseBlockCSCUnsortedRowIndicesWithinColumn() {
		SparseBlockCSC sblock = new SparseBlockCSC(getFixedSparseBlock());
		int[] idxs = new int[]{0, 1, 2, 3, 3, 2}; // unsorted for last col
		setField(sblock, "_indexes", idxs);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, false));
		assertTrue(ex.getMessage().startsWith("Wrong sparse column ordering"));
	}

	@Test
	public void testSparseBlockCSRUnsortedColumnIndicesWithinRow() {
		SparseBlockCSR sblock = new SparseBlockCSR(getFixedSparseBlock());
		int[] idxs = new int[]{0, 1, 2, 3, 3, 2}; // unsorted for last row
		setField(sblock, "_indexes", idxs);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, false));
		assertTrue(ex.getMessage().startsWith("Wrong sparse row ordering"));
	}

	@Test
	public void testSparseBlockDCSRUnsortedColumnIndicesWithinRow() {
		SparseBlockDCSR sblock = new SparseBlockDCSR(getFixedSparseBlock());
		int[] colIdxs = new int[]{0, 1, 2, 3, 3, 2}; // unsorted for last row
		setField(sblock, "_colidx", colIdxs);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, false));
		assertTrue(ex.getMessage().startsWith("Wrong sparse row ordering"));
	}

	@Test
	public void testSparseBlockMCSCUnsortedRowIndicesWithinColumn() {
		SparseBlockMCSC sblock = new SparseBlockMCSC(getFixedSparseBlock());
		int[] indexes = new int[]{3, 2}; // unsorted
		setField(sblock.getCols()[3], "indexes", indexes);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, false));
		assertTrue(ex.getMessage().startsWith("Wrong sparse column ordering"));
	}

	@Test
	public void testSparseBlockMCSRUnsortedColumnIndicesWithinRow() {
		SparseBlockMCSR sblock = new SparseBlockMCSR(getFixedSparseBlock());
		int[] indexes = new int[]{3, 2}; // unsorted
		setField(sblock.getRows()[3], "indexes", indexes);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, false));
		assertTrue(ex.getMessage().startsWith("Wrong sparse row ordering"));
	}

	@Test
	public void testSparseBlockMCSCInvalidIndices() {
		SparseBlockMCSC sblock = new SparseBlockMCSC(getFixedSparseBlock());
		int[] indexes = sblock.getCols()[3].indexes();
		indexes[0] = -1;

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, false));
		assertTrue(ex.getMessage().startsWith("Invalid index"));
	}

	@Test
	public void testSparseBlockMCSRInvalidIndices() {
		SparseBlockMCSR sblock = new SparseBlockMCSR(getFixedSparseBlock());
		int[] indexes = sblock.getRows()[3].indexes();
		indexes[0] = -1;

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, false));
		assertTrue(ex.getMessage().startsWith("Invalid index"));
	}

	@Test
	public void testSparseBlockCOOInvalidValue() {
		runSparseBlockInvalidValueTest(SparseBlock.Type.COO);
	}

	@Test
	public void testSparseBlockCSCInvalidValue() {
		runSparseBlockInvalidValueTest(SparseBlock.Type.CSC);
	}

	@Test
	public void testSparseBlockCSRInvalidValue() {
		runSparseBlockInvalidValueTest(SparseBlock.Type.CSR);
	}

	@Test
	public void testSparseBlockDCSRInvalidValue() {
		runSparseBlockInvalidValueTest(SparseBlock.Type.DCSR);
	}

	@Test
	public void testSparseBlockMCSCInvalidValue() {
		SparseBlockMCSC sblock = new SparseBlockMCSC(getFixedSparseBlock());
		double[] values = sblock.valuesCol(3);
		values[0] = 0;

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, true));
		assertTrue(ex.getMessage().startsWith("The values array should not contain zeros"));
	}

	@Test
	public void testSparseBlockMCSRInvalidValue() {
		SparseBlockMCSR sblock = new SparseBlockMCSR(getFixedSparseBlock());
		double[] values = sblock.values(3);
		values[0] = 0;

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, true));
		assertTrue(ex.getMessage().startsWith("The values array should not contain zeros"));
	}

	@Test
	public void testSparseBlockCOOInvalidRIndex() {
		runSparseBlockInvalidIndexTest(SparseBlock.Type.COO, "_rindexes");
	}

	@Test
	public void testSparseBlockCOOInvalidCIndex() {
		runSparseBlockInvalidIndexTest(SparseBlock.Type.COO, "_cindexes");
	}


	@Test
	public void testSparseBlockCSCInvalidIndex() {
		runSparseBlockInvalidIndexTest(SparseBlock.Type.CSC, "_indexes");
	}

	@Test
	public void testSparseBlockCSRInvalidIndex() {
		runSparseBlockInvalidIndexTest(SparseBlock.Type.CSR, "_indexes");
	}

	@Test
	public void testSparseBlockDCSRInvalidIndex() {
		runSparseBlockInvalidIndexTest(SparseBlock.Type.DCSR, "_colidx");
	}

	@Test
	public void testSparseBlockCOOCapacityExceedsAllowedLimit() {
		SparseBlockCOO sblock = new SparseBlockCOO(3, 50);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(3, 3, 0, false));
		assertTrue(ex.getMessage().startsWith("Capacity is larger than the nnz times a resize factor"));
	}

	@Test
	public void testSparseBlockCSCCapacityExceedsAllowedLimit() {
		SparseBlockCSC sblock = new SparseBlockCSC(3, 3, 50);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(3, 3, 0, false));
		assertTrue(ex.getMessage().startsWith("Capacity is larger than the nnz times a resize factor"));
	}

	@Test
	public void testSparseBlockCSRCapacityExceedsAllowedLimit() {
		SparseBlockCSR sblock = new SparseBlockCSR(3, 50, 0);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(3, 3, 0, false));
		assertTrue(ex.getMessage().startsWith("Capacity is larger than the nnz times a resize factor"));
	}

	@Test
	public void testSparseBlockDCSRCapacityExceedsAllowedLimit() {
		SparseBlockDCSR sblock = new SparseBlockDCSR(3, 50);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(3, 3, 0, false));
		assertTrue(ex.getMessage().startsWith("Capacity is larger than the nnz times a resize factor"));
	}

	@Test
	public void testSparseBlockMCSCCapacityExceedsAllowedLimit() {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 13);

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlockMCSC sblock = new SparseBlockMCSC(srtmp);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(_rows, _cols, 2, true));
		assertTrue(ex.getMessage().startsWith("The capacity is larger than nnz times a resize factor"));
	}

	@Test
	public void testSparseBlockMCSRCapacityExceedsAllowedLimit() {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 13);

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlockMCSR sblock = new SparseBlockMCSR(srtmp);

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(_rows, _cols, 2, true));
		assertTrue(ex.getMessage().startsWith("The capacity is larger than nnz times a resize factor"));
	}

	private void runSparseBlockValidTest(SparseBlock.Type btype)  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 13);

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);

		assertTrue("should pass checkValidity", sblock.checkValidity(_rows, _cols, sblock.size(), true));
	}

	private void runSparseBlockInvalidDimensionsTest(SparseBlock sblock) {
		RuntimeException ex1 = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(-1, 1, 0, false));
		assertTrue(ex1.getMessage().startsWith("Invalid block dimensions"));

		RuntimeException ex2 = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(1, -1, 0, false));
		assertTrue(ex2.getMessage().startsWith("Invalid block dimensions"));
	}

	private void runSparseBlockInvalidIndexTest(SparseBlock.Type btype, String indexName)  {
		SparseBlock srtmp = getFixedSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);

		int[] indexes = (int[]) getField(sblock, indexName);
		indexes[0] = -1;

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, true));
		assertTrue(ex.getMessage().startsWith("Invalid index at pos"));
	}

	private void runSparseBlockInvalidValueTest(SparseBlock.Type btype)  {
		SparseBlock srtmp = getFixedSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);

		double[] values = (double[]) getField(sblock, "_values");
		values[0] = 0;

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, true));
		assertTrue(ex.getMessage().startsWith("The values array should not contain zeros"));
	}

	private void checkValidityFailsWhenArrayLengthIsTemporarilyModified(SparseBlock sblock, String name, Object value){
		Object old = getField(sblock, name);
		setField(sblock, name, value);
		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(4, 4, 6, false));
		assertEquals("Incorrect array lengths.", ex.getMessage());
		setField(sblock, name, old);
	}

	private SparseBlock getFixedSparseBlock(){
		double[][] A = new double[][] {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 1}, {0, 0, 1, 1}};
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		return mbtmp.getSparseBlock();
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

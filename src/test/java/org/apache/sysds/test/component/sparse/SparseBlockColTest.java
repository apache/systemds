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
import org.apache.sysds.runtime.data.SparseBlockCSC;
import org.apache.sysds.runtime.data.SparseBlockMCSC;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.data.SparseRowScalar;
import org.apache.sysds.runtime.data.SparseRowVector;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

public class SparseBlockColTest extends AutomatedTestBase
{
	private final static int _rows = 324;
	private final static int _cols = 132;
	private final static double _sparsity = 0.3;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockCSCGetReset()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockColWrapper b = wrap(new SparseBlockCSC(mbtmp.getSparseBlock()));
		runSparseBlockGetResetTest(b, SparseBlock.Type.CSC);
	}

	@Test
	public void testSparseBlockMCSCGetReset()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockColWrapper b = wrap(new SparseBlockMCSC(mbtmp.getSparseBlock()));
		runSparseBlockGetResetTest(b, SparseBlock.Type.MCSC);
	}

	@Test
	public void testSparseBlockCSCSetSort()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockColWrapper b = wrap(new SparseBlockCSC(mbtmp.getSparseBlock()));
		SparseRow[] cols = (new SparseBlockMCSC(mbtmp.getSparseBlock())).getCols();
		runSparseBlockSetSortTest(b, cols);
	}

	@Test
	public void testSparseBlockMCSCSetSort()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockColWrapper b = wrap(new SparseBlockMCSC(mbtmp.getSparseBlock()));
		SparseRow[] cols = (new SparseBlockMCSC(mbtmp.getSparseBlock())).getCols();
		runSparseBlockSetSortTest(b, cols);
	}

	@Test
	public void testSparseBlockCSCSetDelIdxRange()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockColWrapper b = wrap(new SparseBlockCSC(mbtmp.getSparseBlock()));
		SparseRow[] cols = (new SparseBlockMCSC(mbtmp.getSparseBlock())).getCols();
		runSparseBlockSetDelIdxRangeTest(b, cols);
	}

	@Test
	public void testSparseBlockMCSCSetDelIdxRange()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockColWrapper b = wrap(new SparseBlockMCSC(mbtmp.getSparseBlock()));
		SparseRow[] cols = (new SparseBlockMCSC(mbtmp.getSparseBlock())).getCols();
		runSparseBlockSetDelIdxRangeTest(b, cols);
	}

	private void runSparseBlockGetResetTest(SparseBlockColWrapper sblock, SparseBlock.Type btype)  {
		int c = _cols/3;
		SparseRow col = sblock.getCol(c);
		int size = sblock.sizeCol(c);
		Assert.assertEquals(col.size(), size);

		sblock.resetCol(c);
		col = sblock.getCol(c);
		size = sblock.sizeCol(c);
		Assert.assertEquals(0, size);
		Assert.assertTrue(col.isEmpty());
		if(btype == SparseBlock.Type.CSC) Assert.assertTrue(col instanceof SparseRowScalar);

		// nothing changes
		SparseBlockColWrapper sblock2 = sblock.copy();
		sblock.resetCol(c);
		SparseRow col2 = sblock.getCol(c);
		Assert.assertArrayEquals(col.indexes(), col2.indexes());
		Assert.assertArrayEquals(col.values(), col2.values(), 0);
		Assert.assertEquals(sblock.getObject(), sblock2.getObject());
	}

	private void runSparseBlockSetSortTest(SparseBlockColWrapper sblock, SparseRow[] cols)  {
		int c = _cols/3;
		SparseRow col = cols[c];
		double[] values = col.values().clone();
		int[] indexes = col.indexes().clone();
		int size = col.size();

		// reverse
		for (int i = 0; i < size/2; i++) {
			double t = values[i];
			values[i] = values[size-1-i];
			values[size-1-i] = t;
			int t2 = indexes[i];
			indexes[i] = indexes[size-1-i];
			indexes[size-1-i] = t2;
		}
		Assert.assertFalse(Arrays.equals(col.values(), values));
		Assert.assertFalse(Arrays.equals(col.indexes(), indexes));

		SparseRow col2 = new SparseRowVector(values, indexes);
		sblock.resetCol(c);
		sblock.setCol(c, col2, true);
		Assert.assertArrayEquals(col2.indexes(), sblock.getCol(c).indexes());
		Assert.assertArrayEquals(col2.values(), sblock.getCol(c).values(), 0);

		int nnz = (int) ((SparseBlock) sblock.getObject()).size();
		int rlen = ((SparseBlock) sblock.getObject()).numRows();
		int clen = cols.length;
		RuntimeException ex = Assert.assertThrows(RuntimeException.class,
			() -> ((SparseBlock) sblock.getObject()).checkValidity(rlen, clen, nnz, true));
		Assert.assertTrue(ex.getMessage().startsWith("Wrong sparse column ordering"));

		sblock.sortCol(c);
		Assert.assertTrue(((SparseBlock)sblock.getObject()).checkValidity(rlen, clen, nnz, true));
		Assert.assertArrayEquals(col.indexes(), sblock.getCol(c).indexes());
		Assert.assertArrayEquals(col.values(), sblock.getCol(c).values(), 0);
	}

	private void runSparseBlockSetDelIdxRangeTest(SparseBlockColWrapper sblock, SparseRow[] cols)  {
		int c = _cols/3;
		int rl = _rows/4;
		int ru = _rows/2;

		SparseRow[] cols2 = Arrays.copyOf(cols, cols.length);
		double[] v = getRandomMatrix(1, _rows, -10, 10, 1, 1234)[0];
		for(int i=0; i<rl; i++) v[i] = cols[c].get(i);
		cols2[c] = new SparseRowVector(v);
		SparseBlock sblock2 = new SparseBlockMCSC(cols2, false, _rows);

		sblock.setIndexRangeCol(c, rl, _rows, v, rl, _rows-rl);
		Assert.assertEquals(sblock2, sblock.getObject());

		sblock.deleteIndexRangeCol(c, rl, ru);
		for(int i=rl; i<ru; i++) cols2[c].set(i, 0);
		Assert.assertEquals(sblock2, sblock.getObject());

		sblock.deleteIndexRangeCol(c, rl, _rows+1);
		for(int i=ru; i<_rows; i++) cols2[c].set(i, 0);
		Assert.assertEquals(sblock2, sblock.getObject());
	}

	private interface SparseBlockColWrapper {
		SparseRow getCol(int c);
		void setCol(int c, SparseRow col, boolean deep);
		void setIndexRangeCol(int c, int rl, int ru, double[] v, int vix, int vlen);
		void deleteIndexRangeCol(int c, int rl, int ru);
		int sizeCol(int c);
		void sortCol(int c);
		void resetCol(int c);
		SparseBlockColWrapper copy();
		Object getObject();
	}

	private SparseBlockColWrapper wrap(SparseBlockCSC b) {
		return new SparseBlockColWrapper() {
			@Override
			public SparseRow getCol(int c) { return b.getCol(c); }

			@Override
			public void setCol(int c, SparseRow col, boolean deep) {
				b.setCol(c, col, deep); }

			@Override
			public void setIndexRangeCol(int c, int rl, int ru, double[] v, int vix, int vlen){
				b.setIndexRangeCol(c, rl, ru, v, vix, vlen);
			}

			@Override
			public void deleteIndexRangeCol(int c, int rl, int ru){
				b.deleteIndexRangeCol(c, rl, ru);
			}

			@Override
			public int sizeCol(int c) { return b.sizeCol(c); }

			@Override
			public void sortCol(int c) { b.sortCol(c); }

			@Override
			public void resetCol(int c) { b.resetCol(c); }

			@Override
			public SparseBlockColWrapper copy() { return wrap(new SparseBlockCSC(b)); }

			@Override
			public Object getObject() { return b; }
		};
	}

	private SparseBlockColWrapper wrap(SparseBlockMCSC b) {
		return new SparseBlockColWrapper() {
			@Override
			public SparseRow getCol(int c) { return b.getCol(c); }

			@Override
			public void setCol(int c, SparseRow col, boolean deep) {
				b.setCol(c, col, deep); }

			@Override
			public void setIndexRangeCol(int c, int rl, int ru, double[] v, int vix, int vlen){
				b.setIndexRangeCol(c, rl, ru, v, vix, vlen);
			}

			@Override
			public void deleteIndexRangeCol(int c, int rl, int ru){
				b.deleteIndexRangeCol(c, rl, ru);
			}

			@Override
			public int sizeCol(int c) { return b.sizeCol(c); }

			@Override
			public void sortCol(int c) { b.sortCol(c); }

			@Override
			public void resetCol(int c) { b.resetCol(c, 0, 0); }

			@Override
			public SparseBlockColWrapper copy() { return wrap(new SparseBlockMCSC(b)); }

			@Override
			public Object getObject() { return b; }
		};
	}
}

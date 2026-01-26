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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;


public class SparseBlockInitializationTest extends AutomatedTestBase
{
	private final static int _rows = 324;
	private final static int _cols = 132;
	private final static double _sparsity = 0.22;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockCreationCOO() {
		runSparseBlockCreationTest(SparseBlock.Type.COO);
	}

	@Test
	public void testSparseBlockCreationCSC() {
		runSparseBlockCreationTest(SparseBlock.Type.CSC);
	}

	@Test
	public void testSparseBlockCreationCSR() {
		runSparseBlockCreationTest(SparseBlock.Type.CSR);
	}

	@Test
	public void testSparseBlockCreationDCSR() {
		runSparseBlockCreationTest(SparseBlock.Type.DCSR);
	}

	@Test
	public void testSparseBlockCreationMCSC() {
		runSparseBlockCreationTest(SparseBlock.Type.MCSC);
	}

	@Test
	public void testSparseBlockCreationMCSR() {
		runSparseBlockCreationTest(SparseBlock.Type.MCSR);
	}

	private void runSparseBlockCreationTest(SparseBlock.Type type) {
		SparseBlock sblock = SparseBlockFactory.createSparseBlock(type, _cols);
		assertEquals(sblock.getSparseBlockType(), type);
	}

	@Test
	public void testSparseBlockCOOInitCapacity()  {
		int init_capacity = 4;
		SparseBlockCOO sblock = new SparseBlockCOO(_cols);
		assertEquals("INIT_CAPACITY should be 4", init_capacity, sblock.values(1).length);
	}

	@Test
	public void testSparseBlockCOORows()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock sblock = new SparseBlockCOO(mbtmp.getSparseBlock());

		int totalNnz = 0;
		int rows = A.length;
		SparseRow[] sparseRows = new SparseRow[rows];

		for (int i = 0; i < rows; i++) {
			SparseRow srv = new SparseRowVector(A[i]);
			sparseRows[i] = srv;
			totalNnz += srv.size();
		}

		SparseBlockCOO sblock2 = new SparseBlockCOO(sparseRows, totalNnz);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockCOORowsValuesIndexes()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock sblock = new SparseBlockCOO(mbtmp.getSparseBlock());

		int totalNnz = 0;
		int rows = A.length;
		SparseRow[] sparseRows = new SparseRow[rows];

		for (int i = 0; i < rows; i++) {
			int[] indexes = new int[A[i].length];
			for (int j = 0; j < A[i].length; j++) indexes[j] = j;
			SparseRow srv = new SparseRowVector(A[i], indexes);
			srv.compact();
			sparseRows[i] = srv;
			totalNnz += srv.size();
		}

		SparseBlockCOO sblock2 = new SparseBlockCOO(sparseRows, totalNnz);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockCSCInitCapacity()  {
		int rlen = 4;
		int clen = 5;
		int capacity = 4;
		SparseBlockCSC sblock = new SparseBlockCSC(rlen, clen, capacity);

		assertEquals("num rows should be equal to rlen", rlen, sblock.numRows());
		assertEquals("length ptr should be equal to clen+1", clen+1, sblock.colPointers().length);
		assertEquals("length values should be equal to capacity", capacity, sblock.valuesCol(0).length);
		assertEquals("length indexes should be equal to capacity", capacity, sblock.indexesCol(0).length);
	}

	@Test
	public void testSparseBlockCSCInitPointer()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockCSC sblock = new SparseBlockCSC(mbtmp.getSparseBlock());

		int[] colPtr = sblock.colPointers();
		int[] rowInd = sblock.indexesCol(0);
		double[] values = sblock.valuesCol(0);
		int nnz = sblock.sizeCol(0);
		SparseBlockCSC sblock2 = new SparseBlockCSC(colPtr, rowInd, values, nnz);

		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockCSCInitMSCS()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock sblock = new SparseBlockMCSC(mbtmp.getSparseBlock());

		SparseBlockCSC sblock2 = new SparseBlockCSC(sblock);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockCSCInitCols()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockMCSC sblock = new SparseBlockMCSC(mbtmp.getSparseBlock());

		SparseRow[] cols = sblock.getCols();
		int totalNnz = (int) sblock.size();

		SparseBlock sblock2 = new SparseBlockCSC(cols, totalNnz);
		assertEquals(sblock, sblock2);

	}

	@Test
	public void testSparseBlockCSCInitRowColInd()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockCSC sblock = new SparseBlockCSC(mbtmp.getSparseBlock());

		int[] ptr = sblock.colPointers();
		int[] rowInd = sblock.indexesCol(0);
		double[] values = sblock.valuesCol(0);

		int clen = ptr.length-1;
		int[] colInd = new int[rowInd.length];
		for(int i=0; i<clen; i++) {
			for(int j=ptr[i]; j<ptr[i+1]; j++) {
				colInd[j] = i;
			}
		}

		SparseBlock sblock2 = new SparseBlockCSC(clen, rowInd, colInd, values);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockCSCInitUltraSparse() throws Exception {
		double ultraSparsity = 0.001;
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, ultraSparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockCSC sblock = new SparseBlockCSC(mbtmp.getSparseBlock());

		// stream of ijv triples
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(baos);
		int nnz = 0;
		for (int c = 0; c < _cols; c++) {
			for (int r = 0; r < _rows; r++) {
				double v = A[r][c];
				if (v != 0) {
					dos.writeInt(r);
					dos.writeInt(c);
					dos.writeDouble(v);
					nnz++;
				}
			}
		}
		dos.close();

		SparseBlockCSC sblock2 = new SparseBlockCSC(_rows, _cols);
		DataInputStream dis = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
		sblock2.initUltraSparse(nnz, dis);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockCSCInitSparse() throws Exception {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockCSC sblock = new SparseBlockCSC(mbtmp.getSparseBlock());

		// ijv-stream in CSC order
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(baos);
		int nnz = 0;
		for (int c = 0; c < _cols; c++) {
			int lnnz = 0;
			for (int r = 0; r < _rows; r++) {
				if (A[r][c] != 0)
					lnnz++;
			}
			dos.writeInt(lnnz);
			nnz += lnnz;

			for (int r = 0; r < _rows; r++) {
				double v = A[r][c];
				if (v != 0) {
					dos.writeInt(r);
					dos.writeDouble(v);
				}
			}
		}
		dos.close();

		SparseBlockCSC sblock2 = new SparseBlockCSC(_rows, _cols);
		DataInputStream dis = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
		sblock2.initSparse(_cols, nnz, dis);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockCSRInitRows()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockMCSR sblock = new SparseBlockMCSR(mbtmp.getSparseBlock());

		SparseRow[] rows = sblock.getRows();
		int totalNnz = (int) sblock.size();

		SparseBlock sblock2 = new SparseBlockCSR(rows, totalNnz);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockCSRInitRowColInd()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockCSR sblock = new SparseBlockCSR(mbtmp.getSparseBlock());

		int[] ptr = sblock.rowPointers();
		int[] colInd = sblock.indexes();
		double[] values = sblock.values();

		int rlen = ptr.length-1;
		int[] rowInd = new int[colInd.length];
		for(int i=0; i<rlen; i++) {
			for(int j=ptr[i]; j<ptr[i+1]; j++) {
				rowInd[j] = i;
			}
		}

		SparseBlock sblock2 = new SparseBlockCSR(rlen, rowInd, colInd, values);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockDCSRInitCapacity()  {
		int init_capacity = 4;
		SparseBlockDCSR sblock = new SparseBlockDCSR(_rows);
		assertEquals("INIT_CAPACITY should be 4", init_capacity, sblock.values(1).length);
	}

	@Test
	public void testSparseBlockDCSRInitRowColInd()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		if(!mbtmp.isInSparseFormat()) mbtmp.denseToSparse(true);
		SparseBlockDCSR sblock = new SparseBlockDCSR(mbtmp.getSparseBlock());

		int[] colIdx = sblock.indexes(0);
		double[] values = sblock.values(0);
		int rlen = sblock.numRows();
		int nnz = (int) sblock.size();
		int nnzr = 0;

		int end = 0;
		int[] rowIdx = new int[rlen];
		int[] rowPtr = new int[rlen+1];
		for(int i=0, j=0; i<rlen; i++) {
			if(sblock.size(i) != 0){
				nnzr++;
				end += sblock.size(i);
				rowIdx[j] = i;
				rowPtr[j+1] = end;
				j++;
			}
		}

		SparseBlock sblock2 = new SparseBlockDCSR(rowIdx, rowPtr, colIdx, values, rlen, nnz, nnzr);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockDCSRInitCOO()  {
		testSparseBlockDCSRInitFromSparseBlock(SparseBlock.Type.COO);
	}

	@Test
	public void testSparseBlockDCSRInitCSC()  {
		testSparseBlockDCSRInitFromSparseBlock(SparseBlock.Type.CSC);
	}

	@Test
	public void testSparseBlockDCSRInitMCSC()  {
		testSparseBlockDCSRInitFromSparseBlock(SparseBlock.Type.MCSC);
	}

	@Test
	public void testSparseBlockDCSRInitMCSR()  {
		testSparseBlockDCSRInitFromSparseBlock(SparseBlock.Type.MCSR);
	}

	public void testSparseBlockDCSRInitFromSparseBlock(SparseBlock.Type btype)  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = new SparseBlockDCSR(srtmp);

		SparseBlock sblock2 = SparseBlockFactory.copySparseBlock(btype, srtmp, true);
		SparseBlock sblock3 = new SparseBlockDCSR(sblock2);
		assertEquals(sblock, sblock3);
	}

	@Test
	public void testSparseBlockMCSCInitMCSCOriginalColNull()  {
		double ultraSparsity = 0.001;
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, ultraSparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock sblock = new SparseBlockMCSC(mbtmp.getSparseBlock());

		SparseBlock sblock2 = new SparseBlockMCSC(sblock);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockMCSCInitMCSRNoClenInferred()  {
		double ultraSparsity = 0.001;
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, ultraSparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock sblock = new SparseBlockMCSR(mbtmp.getSparseBlock());

		SparseBlock sblock2 = new SparseBlockMCSC(sblock);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockMCSCInitMCSRClenInferred()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock sblock = new SparseBlockMCSR(mbtmp.getSparseBlock());

		SparseBlock sblock2 = new SparseBlockMCSC(sblock, _cols);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockMCSCInitCSC()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock sblock = new SparseBlockCSC(mbtmp.getSparseBlock());

		SparseBlock sblock2 = new SparseBlockMCSC(sblock);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockMCSCInitColsDeep()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockMCSC sblock = new SparseBlockMCSC(mbtmp.getSparseBlock());

		SparseRow[] cols = sblock.getCols();
		int rlen = sblock.numRows();

		SparseBlock sblock2 = new SparseBlockMCSC(cols, true, rlen);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockMCSCInitColsNonDeep()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockMCSC sblock = new SparseBlockMCSC(mbtmp.getSparseBlock());

		SparseRow[] cols = sblock.getCols();
		int rlen = sblock.numRows();

		SparseBlock sblock2 = new SparseBlockMCSC(cols, false, rlen);
		assertEquals(sblock, sblock2);
	}

	@Test
	public void testSparseBlockMCSCInitClen()  {
		int clen = _cols;
		SparseBlockMCSC sblock = new SparseBlockMCSC(clen);
		assertEquals(clen, sblock.numCols());
	}

	@Test
	public void testSparseBlockMCSRInitRows()  {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 1234);
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlockMCSR sblock = new SparseBlockMCSR(mbtmp.getSparseBlock());

		SparseRow[] rows = sblock.getRows();

		SparseBlockMCSR sblock2 = new SparseBlockMCSR(rows, true);
		assertEquals(sblock, sblock2);
		assertNotSame(sblock.getRows(), sblock2.getRows());
	}

	@Test
	public void testSparseBlockCSRInitSize()  {
		int rlen = 3;
		int capacity = 7;
		int size = 2;
		SparseBlockCSR sblock = new SparseBlockCSR(rlen, capacity, size);
		sblock.append(0, 1, 1.0);
		sblock.append(0, 3, 3.0);
		sblock.compact();
		assertEquals("size should be 2", 2, sblock.size());
	}
}

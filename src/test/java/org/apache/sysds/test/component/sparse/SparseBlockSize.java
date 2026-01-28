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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCOO;
import org.apache.sysds.runtime.data.SparseBlockCSC;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockDCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSC;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

/**
 * This is a sparse matrix block component test for sparse block size 
 * functionality (nnz). In order to achieve broad coverage, we test 
 * against different overloaded versions of size as well as different 
 * sparsity values.
 * 
 */
public class SparseBlockSize extends AutomatedTestBase 
{
	private final static int _rows = 324;
	private final static int _cols = 123;
	private final static int _rl = 31;
	private final static int _ru = 100;
	private final static int _cl = 30;
	private final static int _cu = 80;
	private final static double _sparsity1 = 0.12;
	private final static double _sparsity2 = 0.22;
	private final static double _sparsity3 = 0.32;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}
	
	@Test
	public void testSparseBlockMCSR1()  {
		runSparseBlockSizeTest(SparseBlock.Type.MCSR, _sparsity1);
	}
	
	@Test
	public void testSparseBlockMCSR2()  {
		runSparseBlockSizeTest(SparseBlock.Type.MCSR, _sparsity2);
	}
	
	@Test
	public void testSparseBlockMCSR3()  {
		runSparseBlockSizeTest(SparseBlock.Type.MCSR, _sparsity3);
	}
	
	@Test
	public void testSparseBlockCSR1()  {
		runSparseBlockSizeTest(SparseBlock.Type.CSR, _sparsity1);
	}
	
	@Test
	public void testSparseBlockCSR2()  {
		runSparseBlockSizeTest(SparseBlock.Type.CSR, _sparsity2);
	}
	
	@Test
	public void testSparseBlockCSR3()  {
		runSparseBlockSizeTest(SparseBlock.Type.CSR, _sparsity3);
	}
	
	@Test
	public void testSparseBlockCOO1()  {
		runSparseBlockSizeTest(SparseBlock.Type.COO, _sparsity1);
	}
	
	@Test
	public void testSparseBlockCOO2()  {
		runSparseBlockSizeTest(SparseBlock.Type.COO, _sparsity2);
	}
	
	@Test
	public void testSparseBlockCOO3()  {
		runSparseBlockSizeTest(SparseBlock.Type.COO, _sparsity3);
	}

	@Test
	public void testSparseBlockDCSR1()  {
		runSparseBlockSizeTest(SparseBlock.Type.DCSR, _sparsity1);
	}

	@Test
	public void testSparseBlockDCSR2()  {
		runSparseBlockSizeTest(SparseBlock.Type.DCSR, _sparsity2);
	}

	@Test
	public void testSparseBlockDCSR3()  {
		runSparseBlockSizeTest(SparseBlock.Type.DCSR, _sparsity3);
	}

	@Test
	public void testSparseBlockMCSC1(){
		runSparseBlockSizeTest(SparseBlock.Type.MCSC, _sparsity1);
	}

	@Test
	public void testSparseBlockMCSC2(){
		runSparseBlockSizeTest(SparseBlock.Type.MCSC, _sparsity2);
	}

	@Test
	public void testSparseBlockMCSC3(){
		runSparseBlockSizeTest(SparseBlock.Type.MCSC, _sparsity3);
	}

	@Test
	public void testSparseBlockCSC1(){
		runSparseBlockSizeTest(SparseBlock.Type.CSC, _sparsity1);
	}

	@Test
	public void testSparseBlockCSC2(){
		runSparseBlockSizeTest(SparseBlock.Type.CSC, _sparsity2);
	}

	@Test
	public void testSparseBlockCSC3(){
		runSparseBlockSizeTest(SparseBlock.Type.CSC, _sparsity3);
	}
	
	@Test
	public void testSparseBlockMCSRFixed1(){
		double[][] A = getFixedData1();
		runSparseBlockSizeTest(A, 0, 4, 0, 6, SparseBlock.Type.MCSR);
	}
	
	@Test
	public void testSparseBlockCSRFixed1(){
		double[][] A = getFixedData1();
		runSparseBlockSizeTest(A, 0, 4, 0, 6, SparseBlock.Type.CSR);
	}
	
	@Test
	public void testSparseBlockCOOFixed1(){
		double[][] A = getFixedData1();
		runSparseBlockSizeTest(A, 0, 4, 0, 6, SparseBlock.Type.COO);
	}

	@Test
	public void testSparseBlockCSCFixed1(){
		double[][] A = getFixedData1();
		runSparseBlockSizeTest(A, 0, 4, 0, 6, SparseBlock.Type.CSC);
	}
	
	@Test
	public void testSparseBlockMCSRFixed2(){
		double[][] A = getFixedData2();
		runSparseBlockSizeTest(A, 0, 4, 2, 4, SparseBlock.Type.MCSR);
	}
	
	@Test
	public void testSparseBlockCSRFixed2(){
		double[][] A = getFixedData2();
		runSparseBlockSizeTest(A, 0, 4, 2, 4, SparseBlock.Type.CSR);
	}

	@Test
	public void testSparseBlockCSCFixed2(){
		double[][] A = getFixedData2();
		runSparseBlockSizeTest(A, 0, 4, 2, 4, SparseBlock.Type.CSC);
	}
	
	@Test
	public void testSparseBlockCOOFixed2(){
		double[][] A = getFixedData2();
		runSparseBlockSizeTest(A, 0, 4, 2, 4, SparseBlock.Type.COO);
	}
	
	@Test
	public void testSparseBlockMCSRFixed3(){
		double[][] A = getFixedData3();
		runSparseBlockSizeTest(A, 0, 4, 3, 3, SparseBlock.Type.MCSR);
	}
	
	@Test
	public void testSparseBlockCSRFixed3(){
		double[][] A = getFixedData3();
		runSparseBlockSizeTest(A, 0, 4, 3, 3, SparseBlock.Type.CSR);
	}

	@Test
	public void testSparseBlockCSCFixed3(){
		double[][] A = getFixedData3();
		runSparseBlockSizeTest(A, 0, 4, 3, 3, SparseBlock.Type.CSC);
	}
	
	@Test
	public void testSparseBlockCOOFixed3(){
		double[][] A = getFixedData3();
		runSparseBlockSizeTest(A, 0, 4, 3, 3, SparseBlock.Type.COO);
	}
	
	private void runSparseBlockSizeTest(SparseBlock.Type btype, double sparsity) {
		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, sparsity, 123); 
		runSparseBlockSizeTest(A, _rl, _ru, _cl, _cu, btype);
	}

	private void runSparseBlockSizeTest(double[][] A,
		int rl, int ru, int cl, int cu, SparseBlock.Type btype)
	{
		try
		{
			//init sparse block
			SparseBlock sblock = null;
			MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
			int rows= mbtmp.getNumRows();
			int cols= mbtmp.getNumColumns();
			if( !mbtmp.isInSparseFormat() )
				mbtmp.denseToSparse(true);
			SparseBlock srtmp = mbtmp.getSparseBlock();
			switch( btype ) {
				case MCSR: sblock = new SparseBlockMCSR(srtmp); break;
				case CSR: sblock = new SparseBlockCSR(srtmp); break;
				case COO: sblock = new SparseBlockCOO(srtmp); break;
				case DCSR: sblock = new SparseBlockDCSR(srtmp); break;
				case MCSC: sblock = new SparseBlockMCSC(srtmp, cols); break;
				case CSC: sblock = new SparseBlockCSC(srtmp, cols); break;
			}
			
			//prepare summary statistics nnz
			int[] rnnz = new int[rows];
			int nnz = 0;
			int nnz2 = 0;
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<cols; j++ ) {
					rnnz[i] += (A[i][j]!=0) ? 1 : 0;
					nnz2 += (i >= rl && j >= cl && i < ru && j < cu && A[i][j] != 0) ? 1 : 0;
				}
				nnz += rnnz[i];
			}

			//check full block nnz
			if(nnz != sblock.size())
				Assert.fail("Wrong number of non-zeros: " + sblock.size() + ", expected: " + nnz);

			//check row nnz
			for(int i = 0; i < rows; i++)
				if(sblock.size(i) != rnnz[i])
					Assert.fail("Wrong number of row non-zeros (" + i + "): " + sblock.size(i) + ", expected: " + rnnz[i]);

			//check two row nnz
			for(int i = 1; i < rows; i++)
				if(sblock.size(i - 1, i + 1) != rnnz[i - 1] + rnnz[i]) {
					Assert.fail("Wrong number of row block non-zeros (" + (i - 1) + "," + (i + 1) + "): " +
						sblock.size(i - 1, i + 1) + ", expected: " + rnnz[i - 1] + rnnz[i]);
				}

			//check index range nnz
			if(sblock.size(rl, ru, cl, cu) != nnz2)
				Assert.fail("Wrong number of range non-zeros: " + sblock.size(rl, ru, cl, cu) + ", expected: " + nnz2);

		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
	
	private double[][] getFixedData1(){
		return new double[][]
			{{10, 20, 0, 0, 0, 0},
			{0, 30, 0, 40, 0, 0},
			{0, 0, 50, 60, 70, 0},
			{0, 0, 0, 0, 0, 80}};
	}
	
	private double[][] getFixedData2(){
		return new double[][]
			{{10, 10, 0, 0, 0, 20},
			{10, 10, 0, 0, 0, 20},
			{10, 10, 0, 0, 0, 20},
			{10, 10, 0, 0, 0, 20}};
	}
	
	private double[][] getFixedData3(){
		return new double[][]
			{{10, 10, 0, 15, 0, 20},
			{10, 10, 0, 15, 0, 20},
			{10, 10, 0, 15, 0, 20},
			{10, 10, 0, 15, 0, 20}};
	}
}

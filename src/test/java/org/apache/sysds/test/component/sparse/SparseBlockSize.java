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

import org.apache.sysds.runtime.data.*;
import org.junit.Assert;
import org.junit.Test;
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
	private final static int rows = 324;
	private final static int cols = 123;
	private final static int rl = 31;
	private final static int ru = 100;
	private final static int cl = 30;
	private final static int cu = 80;
	private final static double sparsity1 = 0.12;
	private final static double sparsity2 = 0.22;
	private final static double sparsity3 = 0.32;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}
	
	@Test
	public void testSparseBlockMCSR1()  {
		runSparseBlockSizeTest(SparseBlock.Type.MCSR, sparsity1);
	}
	
	@Test
	public void testSparseBlockMCSR2()  {
		runSparseBlockSizeTest(SparseBlock.Type.MCSR, sparsity2);
	}
	
	@Test
	public void testSparseBlockMCSR3()  {
		runSparseBlockSizeTest(SparseBlock.Type.MCSR, sparsity3);
	}
	
	@Test
	public void testSparseBlockCSR1()  {
		runSparseBlockSizeTest(SparseBlock.Type.CSR, sparsity1);
	}
	
	@Test
	public void testSparseBlockCSR2()  {
		runSparseBlockSizeTest(SparseBlock.Type.CSR, sparsity2);
	}
	
	@Test
	public void testSparseBlockCSR3()  {
		runSparseBlockSizeTest(SparseBlock.Type.CSR, sparsity3);
	}
	
	@Test
	public void testSparseBlockCOO1()  {
		runSparseBlockSizeTest(SparseBlock.Type.COO, sparsity1);
	}
	
	@Test
	public void testSparseBlockCOO2()  {
		runSparseBlockSizeTest(SparseBlock.Type.COO, sparsity2);
	}
	
	@Test
	public void testSparseBlockCOO3()  {
		runSparseBlockSizeTest(SparseBlock.Type.COO, sparsity3);
	}

	@Test
	public void testSparseBlockDCSR1()  {
		runSparseBlockSizeTest(SparseBlock.Type.DCSR, sparsity1);
	}

	@Test
	public void testSparseBlockDCSR2()  {
		runSparseBlockSizeTest(SparseBlock.Type.DCSR, sparsity2);
	}

	@Test
	public void testSparseBlockDCSR3()  {
		runSparseBlockSizeTest(SparseBlock.Type.DCSR, sparsity3);
	}

	@Test
	public void testSparseBlockMCSC1(){
		runSparseBlockSizeTest(SparseBlock.Type.MCSC, sparsity1);
	}

	@Test
	public void testSparseBlockMCSC2(){
		runSparseBlockSizeTest(SparseBlock.Type.MCSC, sparsity2);
	}

	@Test
	public void testSparseBlockMCSC3(){
		runSparseBlockSizeTest(SparseBlock.Type.MCSC, sparsity3);
	}

	private void runSparseBlockSizeTest( SparseBlock.Type btype, double sparsity)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 123); 
			
			//init sparse block
			SparseBlock sblock = null;
			MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
			SparseBlock srtmp = mbtmp.getSparseBlock();
			switch( btype ) {
				case MCSR: sblock = new SparseBlockMCSR(srtmp); break;
				case CSR: sblock = new SparseBlockCSR(srtmp); break;
				case COO: sblock = new SparseBlockCOO(srtmp); break;
				case DCSR: sblock = new SparseBlockDCSR(srtmp); break;
				case MCSC: sblock = new SparseBlockMCSC(srtmp); break;
			}
			
			//prepare summary statistics nnz
			int[] rnnz = new int[rows];
			int[] cnnz = new int[cols];
			int nnz = 0;
			int nnz2 = 0;
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<cols; j++ ) {
					cnnz[j] += (A[i][j]!=0) ? 1 : 0;
					rnnz[i] += (A[i][j]!=0) ? 1 : 0;
					nnz2 += (i>=rl && j>=cl && i<ru && j<cu && A[i][j]!=0) ? 1 : 0;
				}
				nnz += rnnz[i];
			}
			
			//check full block nnz
			if( nnz != sblock.size() )
				Assert.fail("Wrong number of non-zeros: "+sblock.size()+", expected: "+nnz);

			//check row nnz
			//for MCSC we check columns
			if(sblock instanceof SparseBlockMCSC) {
				for(int i = 0; i < cols; i++)
					if(sblock.size(i) != cnnz[i]) {
						Assert.fail("Wrong number of column non-zeros (" + i + "): " + sblock.size(i) + ", expected: " +
							cnnz[i]);
					}
			}
			else {
				for(int i = 0; i < rows; i++)
					if(sblock.size(i) != rnnz[i]) {
						Assert.fail(
							"Wrong number of row non-zeros (" + i + "): " + sblock.size(i) + ", expected: " + rnnz[i]);
					}
			}

			//check for two column nnz
			if(sblock instanceof SparseBlockMCSC) {
				for(int i = 1; i < cols; i++)
					if(sblock.size(i - 1, i + 1) != cnnz[i - 1] + cnnz[i]) {
						Assert.fail("Wrong number of column block non-zeros (" + (i - 1) + "," + (i + 1) + "): " +
							sblock.size(i - 1, i + 1) + ", expected: " + cnnz[i - 1] + cnnz[i]);
					}
			}
			else {
				//check two row nnz
				for(int i = 1; i < rows; i++)
					if(sblock.size(i - 1, i + 1) != rnnz[i - 1] + rnnz[i]) {
						Assert.fail("Wrong number of row block non-zeros (" + (i - 1) + "," + (i + 1) + "): " +
							sblock.size(i - 1, i + 1) + ", expected: " + rnnz[i - 1] + rnnz[i]);
					}
			}
			
			//check index range nnz
			if( sblock.size(rl, ru, cl, cu) != nnz2 )
				Assert.fail("Wrong number of range non-zeros: " +
						sblock.size(rl, ru, cl, cu)+", expected: "+nnz2);

		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}

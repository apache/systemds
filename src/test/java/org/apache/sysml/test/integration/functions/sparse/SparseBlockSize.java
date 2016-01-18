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

package org.apache.sysml.test.integration.functions.sparse;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlockCOO;
import org.apache.sysml.runtime.matrix.data.SparseBlockCSR;
import org.apache.sysml.runtime.matrix.data.SparseBlockMCSR;
import org.apache.sysml.runtime.matrix.data.SparseRow;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

/**
 * This is a sparse matrix block component test for sparse block size 
 * functionality (nnz). In order to achieve broad coverage, we test 
 * against different overloaded versions of size as well as different 
 * sparsity values.
 * 
 */
public class SparseBlockSize extends AutomatedTestBase 
{
	private final static int rows = 762;
	private final static int cols = 649;
	private final static int rl = 31;
	private final static int ru = 345;
	private final static int cl = 345;
	private final static int cu = 525;
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
		
	/**
	 * 
	 * @param btype
	 * @param sparsity
	 */
	private void runSparseBlockSizeTest( SparseBlock.Type btype, double sparsity)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 123); 
			
			//init sparse block
			SparseBlock sblock = null;
			MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
			SparseRow[] srtmp = mbtmp.getSparseBlock();			
			switch( btype ) {
				case MCSR: sblock = new SparseBlockMCSR(srtmp,true); break;
				case CSR: sblock = new SparseBlockCSR(srtmp, (int)mbtmp.getNonZeros()); break;
				case COO: sblock = new SparseBlockCOO(srtmp, (int)mbtmp.getNonZeros()); break;
			}
			
			//prepare summary statistics nnz
			int[] rnnz = new int[rows]; 
			int nnz = 0;
			int nnz2 = 0;
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<cols; j++ ) {
					rnnz[i] += (A[i][j]!=0) ? 1 : 0;
					nnz2 += (i>=rl && j>=cl && i<ru && j<cu && A[i][j]!=0) ? 1 : 0;
				}
				nnz += rnnz[i];
			}
			
			//check full block nnz
			if( nnz != sblock.size() )
				Assert.fail("Wrong number of non-zeros: "+sblock.size()+", expected: "+nnz);
		
			//check row nnz
			for( int i=0; i<rows; i++ )
				if( sblock.size(i) != rnnz[i] ) {
					Assert.fail("Wrong number of row non-zeros ("+i+"): " +
							sblock.size(i)+", expected: "+rnnz[i]);
				}
			
			//check two row nnz 
			for( int i=1; i<rows; i++ )
				if( sblock.size(i-1,i+1) != rnnz[i-1]+rnnz[i] ) {
					Assert.fail("Wrong number of row block non-zeros ("+(i-1)+","+(i+1)+"): " +
							sblock.size(i-1,i+1)+", expected: "+rnnz[i-1]+rnnz[i]);
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
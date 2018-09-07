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
import org.apache.sysml.runtime.matrix.data.SparseBlockFactory;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

public class SparseBlockMerge extends AutomatedTestBase 
{
	private final static int rows = 1000;
	private final static int cols = 1000;
	private final static double sparsity0 = 0.000005;
	private final static double sparsity1 = 0.001;
	private final static double sparsity2 = 0.01;
	private final static double sparsity3 = 0.1;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testMergeMCSR_MCSR_0()  {
		runSparseBlockMergeTest(SparseBlock.Type.MCSR, SparseBlock.Type.MCSR, sparsity0);
	}
	
	@Test
	public void testMergeMCSR_MCSR_1()  {
		runSparseBlockMergeTest(SparseBlock.Type.MCSR, SparseBlock.Type.MCSR, sparsity1);
	}
	
	@Test
	public void testMergeMCSR_MCSR_2()  {
		runSparseBlockMergeTest(SparseBlock.Type.MCSR, SparseBlock.Type.MCSR, sparsity2);
	}
	
	@Test
	public void testMergeMCSR_MCSR_3()  {
		runSparseBlockMergeTest(SparseBlock.Type.MCSR, SparseBlock.Type.MCSR, sparsity3);
	}
	
	@Test
	public void testMergeMCSR_CSR_0()  {
		runSparseBlockMergeTest(SparseBlock.Type.MCSR, SparseBlock.Type.CSR, sparsity0);
	}
	
	@Test
	public void testMergeMCSR_CSR_1()  {
		runSparseBlockMergeTest(SparseBlock.Type.MCSR, SparseBlock.Type.CSR, sparsity1);
	}
	
	@Test
	public void testMergeMCSR_CSR_2()  {
		runSparseBlockMergeTest(SparseBlock.Type.MCSR, SparseBlock.Type.CSR, sparsity2);
	}
	
	@Test
	public void testMergeMCSR_CSR_3()  {
		runSparseBlockMergeTest(SparseBlock.Type.MCSR, SparseBlock.Type.CSR, sparsity3);
	}
	
	@Test
	public void testMergeMCSR_COO_0()  {
		runSparseBlockMergeTest(SparseBlock.Type.MCSR, SparseBlock.Type.COO, sparsity0);
	}
	
	@Test
	public void testMergeMCSR_COO_1()  {
		runSparseBlockMergeTest(SparseBlock.Type.MCSR, SparseBlock.Type.COO, sparsity1);
	}
	
	@Test
	public void testMergeMCSR_COO_2()  {
		runSparseBlockMergeTest(SparseBlock.Type.MCSR, SparseBlock.Type.COO, sparsity2);
	}
	
	@Test
	public void testMergeMCSR_COO_3()  {
		runSparseBlockMergeTest(SparseBlock.Type.MCSR, SparseBlock.Type.COO, sparsity3);
	}
	
	@Test
	public void testMergeCSR_CSR_0()  {
		runSparseBlockMergeTest(SparseBlock.Type.CSR, SparseBlock.Type.CSR, sparsity0);
	}
	
	@Test
	public void testMergeCSR_CSR_1()  {
		runSparseBlockMergeTest(SparseBlock.Type.CSR, SparseBlock.Type.CSR, sparsity1);
	}
	
	@Test
	public void testMergeCSR_CSR_2()  {
		runSparseBlockMergeTest(SparseBlock.Type.CSR, SparseBlock.Type.CSR, sparsity2);
	}
	
	@Test
	public void testMergeCSR_CSR_3()  {
		runSparseBlockMergeTest(SparseBlock.Type.CSR, SparseBlock.Type.CSR, sparsity3);
	}
	
	@Test
	public void testMergeCSR_MCSR_0()  {
		runSparseBlockMergeTest(SparseBlock.Type.CSR, SparseBlock.Type.MCSR, sparsity0);
	}
	
	@Test
	public void testMergeCSR_MCSR_1()  {
		runSparseBlockMergeTest(SparseBlock.Type.CSR, SparseBlock.Type.MCSR, sparsity1);
	}
	
	@Test
	public void testMergeCSR_MCSR_2()  {
		runSparseBlockMergeTest(SparseBlock.Type.CSR, SparseBlock.Type.MCSR, sparsity2);
	}
	
	@Test
	public void testMergeCSR_MCSR_3()  {
		runSparseBlockMergeTest(SparseBlock.Type.CSR, SparseBlock.Type.MCSR, sparsity3);
	}
	
	@Test
	public void testMergeCSR_COO_0()  {
		runSparseBlockMergeTest(SparseBlock.Type.CSR, SparseBlock.Type.COO, sparsity0);
	}
	
	@Test
	public void testMergeCSR_COO_1()  {
		runSparseBlockMergeTest(SparseBlock.Type.CSR, SparseBlock.Type.COO, sparsity1);
	}
	
	@Test
	public void testMergeCSR_COO_2()  {
		runSparseBlockMergeTest(SparseBlock.Type.CSR, SparseBlock.Type.COO, sparsity2);
	}
	
	@Test
	public void testMergeCSR_COO_3()  {
		runSparseBlockMergeTest(SparseBlock.Type.CSR, SparseBlock.Type.COO, sparsity3);
	}
	
	@Test
	public void testMergeCOO_COO_0()  {
		runSparseBlockMergeTest(SparseBlock.Type.COO, SparseBlock.Type.COO, sparsity0);
	}
	
	@Test
	public void testMergeCOO_COO_1()  {
		runSparseBlockMergeTest(SparseBlock.Type.COO, SparseBlock.Type.COO, sparsity1);
	}
	
	@Test
	public void testMergeCOO_COO_2()  {
		runSparseBlockMergeTest(SparseBlock.Type.COO, SparseBlock.Type.COO, sparsity2);
	}
	
	@Test
	public void testMergeCOO_COO_3()  {
		runSparseBlockMergeTest(SparseBlock.Type.COO, SparseBlock.Type.COO, sparsity3);
	}
	
	@Test
	public void testMergeCOO_MCSR_0()  {
		runSparseBlockMergeTest(SparseBlock.Type.COO, SparseBlock.Type.MCSR, sparsity0);
	}
	
	@Test
	public void testMergeCOO_MCSR_1()  {
		runSparseBlockMergeTest(SparseBlock.Type.COO, SparseBlock.Type.MCSR, sparsity1);
	}
	
	@Test
	public void testMergeCOO_MCSR_2()  {
		runSparseBlockMergeTest(SparseBlock.Type.COO, SparseBlock.Type.MCSR, sparsity2);
	}
	
	@Test
	public void testMergeCOO_MCSR_3()  {
		runSparseBlockMergeTest(SparseBlock.Type.COO, SparseBlock.Type.MCSR, sparsity3);
	}
	
	@Test
	public void testMergeCOO_CSR_0()  {
		runSparseBlockMergeTest(SparseBlock.Type.COO, SparseBlock.Type.CSR, sparsity0);
	}
	
	@Test
	public void testMergeCOO_CSR_1()  {
		runSparseBlockMergeTest(SparseBlock.Type.COO, SparseBlock.Type.CSR, sparsity1);
	}
	
	@Test
	public void testMergeCOO_CSR_2()  {
		runSparseBlockMergeTest(SparseBlock.Type.COO, SparseBlock.Type.CSR, sparsity2);
	}
	
	@Test
	public void testMergeCOO_CSR_3()  {
		runSparseBlockMergeTest(SparseBlock.Type.COO, SparseBlock.Type.CSR, sparsity3);
	}
	
	private void runSparseBlockMergeTest( SparseBlock.Type btype1, SparseBlock.Type btype2, double sparsity)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 1234); 
			double[][] B1 = new double[A.length][];
			double[][] B2 = new double[A.length][];
			for(int i=0; i<A.length; i++) {
				B1[i] = new double[A[i].length];
				B2[i] = new double[A[2].length];
				for(int j=0; j<A[i].length; j++) {
					if( j%2 == 0 )
						B1[i][j] = A[i][j];
					else
						B2[i][j] = A[i][j];
				}
			}
			
			//init sparse block
			MatrixBlock mb1 = DataConverter.convertToMatrixBlock(B1);
			MatrixBlock mb2 = DataConverter.convertToMatrixBlock(B2);
			long nnz = mb1.getNonZeros() + mb2.getNonZeros();
			mb1.setSparseBlock(SparseBlockFactory.copySparseBlock(btype1, mb1.getSparseBlock(), false));
			mb2.setSparseBlock(SparseBlockFactory.copySparseBlock(btype2, mb2.getSparseBlock(), false));
			
			//execute merge
			mb1.merge(mb2, false);
			
			//check for correct number of non-zeros
			if( nnz != mb1.getNonZeros() )
				Assert.fail("Wrong number of non-zeros: "+mb1.getNonZeros()+", expected: "+nnz);
			
			//check correct values
			long count = 0;
			SparseBlock sblock = mb1.getSparseBlock();
			if( sblock != null ) {
				for( int i=0; i<rows; i++) {
					if( sblock.isEmpty(i) ) continue;
					int alen = sblock.size(i);
					int apos = sblock.pos(i);
					int[] aix = sblock.indexes(i);
					double[] avals = sblock.values(i);
					for( int j=0; j<alen; j++ ) {
						if( avals[apos+j] != A[i][aix[apos+j]] )
							Assert.fail("Wrong value returned by scan: "+avals[apos+j]+", expected: "+A[i][apos+aix[j]]);
						count++;
					}
				}
			}
			if( count != nnz )
				Assert.fail("Wrong number of values returned by merge: "+count+", expected: "+nnz);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}

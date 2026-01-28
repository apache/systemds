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
 * This is a sparse matrix block component test for sparse block scan 
 * functionality. In order to achieve broad coverage, we test against 
 * different sparsity values.
 * 
 */
public class SparseBlockScan extends AutomatedTestBase 
{
	private final static int rows = 324;
	private final static int cols = 131;	
	private final static double sparsity1 = 0.09;
	private final static double sparsity2 = 0.19;
	private final static double sparsity3 = 0.29;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockMCSR1Full()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSR, sparsity1);
	}
	
	@Test
	public void testSparseBlockMCSR2Full()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSR, sparsity2);
	}
	
	@Test
	public void testSparseBlockMCSR3Full()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSR, sparsity3);
	}
	
	@Test
	public void testSparseBlockCSR1Full()  {
		runSparseBlockScanTest(SparseBlock.Type.CSR, sparsity1);
	}
	
	@Test
	public void testSparseBlockCSR2Full()  {
		runSparseBlockScanTest(SparseBlock.Type.CSR, sparsity2);
	}
	
	@Test
	public void testSparseBlockCSR3Full()  {
		runSparseBlockScanTest(SparseBlock.Type.CSR, sparsity3);
	}
	
	@Test
	public void testSparseBlockCOO1Full()  {
		runSparseBlockScanTest(SparseBlock.Type.COO, sparsity1);
	}
	
	@Test
	public void testSparseBlockCOO2Full()  {
		runSparseBlockScanTest(SparseBlock.Type.COO, sparsity2);
	}
	
	@Test
	public void testSparseBlockCOO3Full()  {
		runSparseBlockScanTest(SparseBlock.Type.COO, sparsity3);
	}

	@Test
	public void testSparseBlockDCSR1Full()  {
		runSparseBlockScanTest(SparseBlock.Type.DCSR, sparsity1);
	}

	@Test
	public void testSparseBlockDCSR2Full()  {
		runSparseBlockScanTest(SparseBlock.Type.DCSR, sparsity2);
	}

	@Test
	public void testSparseBlockDCSR3Full()  {
		runSparseBlockScanTest(SparseBlock.Type.DCSR, sparsity3);
	}

	@Test
	public void testSparseBlockMCSC1Full()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSC, sparsity1);
	}

	@Test
	public void testSparseBlockMCSC2Full()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSC, sparsity2);
	}

	@Test
	public void testSparseBlockMCSC3Full()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSC, sparsity3);
	}

	@Test
	public void testSparseBlockCSC1Full()  {
		runSparseBlockScanTest(SparseBlock.Type.CSC, sparsity1);
	}

	@Test
	public void testSparseBlockCSC2Full()  {
		runSparseBlockScanTest(SparseBlock.Type.CSC, sparsity2);
	}

	@Test
	public void testSparseBlockCSC3Full()  {
		runSparseBlockScanTest(SparseBlock.Type.CSC, sparsity3);
	}
	
	private void runSparseBlockScanTest( SparseBlock.Type btype, double sparsity)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 1234); 
			
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
				case CSC: sblock = new SparseBlockCSC(srtmp); break;
			}
			
			//check for correct number of non-zeros
			int[] rnnz = new int[rows]; int nnz = 0;
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++)
					rnnz[i] += (A[i][j] != 0) ? 1 : 0;
				nnz += rnnz[i];
			}

			if(nnz != sblock.size())
				Assert.fail("Wrong number of non-zeros: " + sblock.size() + ", expected: " + nnz);

			//check correct isEmpty return
			for(int i = 0; i < rows; i++)
				if(sblock.isEmpty(i) != (rnnz[i] == 0))
					Assert.fail("Wrong isEmpty(row) result for row nnz: " + rnnz[i]);

			//check correct values
			int count = 0;
			for(int i = 0; i < rows; i++) {
				int alen = sblock.size(i);
				int apos = sblock.pos(i);
				int[] aix = sblock.indexes(i);
				double[] avals = sblock.values(i);
				for(int j = 0; j < alen; j++) {
					if(avals[apos + j] != A[i][aix[apos + j]])
						Assert.fail(
							"Wrong value returned by scan: " + avals[apos + j] + ", expected: " + A[i][apos + aix[j]]);
					count++;
				}
			}
			if( count != nnz )
				Assert.fail("Wrong number of values returned by scan: "+count+", expected: "+nnz);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}

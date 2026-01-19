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

import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

/**
 * This is a sparse matrix block component test for sparse block 
 * alignment check functionality. In order to achieve broad coverage, 
 * we test against full/rowwise and different sparsity values.
 * 
 */
public class SparseBlockAlignment extends AutomatedTestBase 
{
	private final static int rows = 324;
	private final static int cols = 132;
	private final static double sparsity1 = 0.09;
	private final static double sparsity2 = 0.19;
	private final static double sparsity3 = 0.29;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockMCSR1Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSR, sparsity1, true);
	}
	
	@Test
	public void testSparseBlockMCSR2Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSR, sparsity2, true);
	}
	
	@Test
	public void testSparseBlockMCSR3Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSR, sparsity3, true);
	}
	
	@Test
	public void testSparseBlockCSR1Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.CSR, sparsity1, true);
	}
	
	@Test
	public void testSparseBlockCSR2Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.CSR, sparsity2, true);
	}
	
	@Test
	public void testSparseBlockCSR3Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.CSR, sparsity3, true);
	}
	
	@Test
	public void testSparseBlockCOO1Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.COO, sparsity1, true);
	}
	
	@Test
	public void testSparseBlockCOO2Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.COO, sparsity2, true);
	}
	
	@Test
	public void testSparseBlockCOO3Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.COO, sparsity3, true);
	}

	@Test
	public void testSparseBlockDCSR1Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.DCSR, sparsity1, true);
	}

	@Test
	public void testSparseBlockDCSR2Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.DCSR, sparsity2, true);
	}

	@Test
	public void testSparseBlockDCSR3Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.DCSR, sparsity3, true);
	}
	
	@Test
	public void testSparseBlockMCSC1Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSC, sparsity1, true);
	}

	@Test
	public void testSparseBlockMCSC2Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSC, sparsity2, true);
	}

	@Test
	public void testSparseBlockMCSC3Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSC, sparsity3, true);
	}

	@Test
	public void testSparseBlockCSC1Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.CSC, sparsity1, true);
	}

	@Test
	public void testSparseBlockCSC2Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.CSC, sparsity2, true);
	}

	@Test
	public void testSparseBlockCSC3Pos()  {
		runSparseBlockScanTest(SparseBlock.Type.CSC, sparsity3, true);
	}

	@Test
	public void testSparseBlockMCSR1Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSR, sparsity1, false);
	}
	
	@Test
	public void testSparseBlockMCSR2Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSR, sparsity2, false);
	}
	
	@Test
	public void testSparseBlockMCSR3Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSR, sparsity3, false);
	}
	
	@Test
	public void testSparseBlockCSR1Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.CSR, sparsity1, false);
	}
	
	@Test
	public void testSparseBlockCSR2Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.CSR, sparsity2, false);
	}
	
	@Test
	public void testSparseBlockCSR3Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.CSR, sparsity3, false);
	}
	
	@Test
	public void testSparseBlockCOO1Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.COO, sparsity1, false);
	}
	
	@Test
	public void testSparseBlockCOO2Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.COO, sparsity2, false);
	}
	
	@Test
	public void testSparseBlockCOO3Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.COO, sparsity3, false);
	}

	@Test
	public void testSparseBlockDCSR1Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.DCSR, sparsity1, false);
	}

	@Test
	public void testSparseBlockDCSR2Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.DCSR, sparsity2, false);
	}

	@Test
	public void testSparseBlockDCSR3Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.DCSR, sparsity3, false);
	}

	@Test
	public void testSparseBlockMCSC1Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSC, sparsity1, false);
	}

	@Test
	public void testSparseBlockMCSC2Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSC, sparsity2, false);
	}

	@Test
	public void testSparseBlockMCSC3Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.MCSC, sparsity3, false);
	}

	@Test
	public void testSparseBlockCSC1Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.CSC, sparsity1, false);
	}

	@Test
	public void testSparseBlockCSC2Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.CSC, sparsity2, false);
	}

	@Test
	public void testSparseBlockCSC3Neg()  {
		runSparseBlockScanTest(SparseBlock.Type.CSC, sparsity3, false);
	}

	private void runSparseBlockScanTest( SparseBlock.Type btype, double sparsity, boolean positive)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 1234); 
			
			//init sparse block
			MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
			SparseBlock srtmp = mbtmp.getSparseBlock();
			SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true, cols);
			
			//init second sparse block and deep copy
			SparseBlock sblock2 = SparseBlockFactory.copySparseBlock(btype, sblock, true, cols);
			
			//modify second block if necessary
			if( !positive ) {
				sblock2.deleteIndexRange(37, 0, cols-1);
				sblock2.deleteIndexRange(38, 0, cols-1);
			}
			
			//check for block comparison
			boolean blockAligned = sblock.isAligned(sblock2);
			if( blockAligned != positive )
				Assert.fail("Wrong block alignment indicated: "+blockAligned+", expected: "+positive);
			
			//check for row comparison
			boolean rowsAligned37 = true;
			boolean rowsAlignedRest = true;
			for( int i=0; i<rows; i++ ) {
				if( i==37 || i==38 )
					rowsAligned37 &= sblock.isAligned(i, sblock2);
				else if( i<37 ) {//CSR/COO different after update pos
					rowsAlignedRest &= sblock.isAligned(i, sblock2);
					if (!sblock.isAligned(i, sblock2)) {
						Assert.fail("Alignment problem at row: " + i + " (" + sblock.size(i) + " vs " + sblock2.size(i) + ")");
					}
				}
			}
			if( rowsAligned37 != positive )
				Assert.fail("Wrong row alignment indicated: "+rowsAligned37+", expected: "+positive);
			if( !rowsAlignedRest )
				Assert.fail("Wrong row alignment rest indicated: false.");

			//init third sparse block with different number of rows
			SparseBlock sblock3 =SparseBlockFactory.createSparseBlock(btype, rows+1);
			if (sblock.isAligned(sblock3)) {
				Assert.fail("Wrong alignment different rows indicated: true.");
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}

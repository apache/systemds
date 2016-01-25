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
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

/**
 * This is a sparse matrix block component test for sparse block 
 * alignment check functionality. In order to achieve broad coverage, 
 * we test against full/rowwise and different sparsity values.
 * 
 */
public class SparseBlockAlignment extends AutomatedTestBase 
{
	private final static int rows = 878;
	private final static int cols = 393;	
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
	
	/**
	 * 
	 * @param btype
	 * @param sparsity
	 * @param positive
	 */
	private void runSparseBlockScanTest( SparseBlock.Type btype, double sparsity, boolean positive)
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
			}
			
			//init second sparse block and deep copy
			SparseBlock sblock2 = null;
			switch( btype ) {
				case MCSR: sblock2 = new SparseBlockMCSR(sblock); break;
				case CSR: sblock2 = new SparseBlockCSR(sblock); break;
				case COO: sblock2 = new SparseBlockCOO(sblock); break;
			}
			
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
				else if( i<37 ) //CSR/COO different after update pos
					rowsAlignedRest &= sblock.isAligned(i, sblock2);
			}
			if( rowsAligned37 != positive )
				Assert.fail("Wrong row alignment indicated: "+rowsAligned37+", expected: "+positive);
			if( !rowsAlignedRest )
				Assert.fail("Wrong row alignment rest indicated: false.");
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
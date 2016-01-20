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
 * This is a sparse matrix block component test for sparse block get
 * first index functionality. In order to achieve broad coverage, we 
 * test against GT, GTE, and LTE as well as different sparsity values.
 * 
 */
public class SparseBlockGetFirstIndex extends AutomatedTestBase 
{
	private final static int rows = 571;
	private final static int cols = 595;
	private final static double sparsity1 = 0.09;
	private final static double sparsity2 = 0.19;
	private final static double sparsity3 = 0.29;
	
	public enum IndexType {
		GT,
		GTE,
		LTE,
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockMCSR1GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSR, sparsity1, IndexType.GT);
	}
	
	@Test
	public void testSparseBlockMCSR2GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSR, sparsity2, IndexType.GT);
	}
	
	@Test
	public void testSparseBlockMCSR3GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSR, sparsity3, IndexType.GT);
	}
	
	@Test
	public void testSparseBlockMCSR1GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSR, sparsity1, IndexType.GTE);
	}
	
	@Test
	public void testSparseBlockMCSR2GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSR, sparsity2, IndexType.GTE);
	}
	
	@Test
	public void testSparseBlockMCSR3GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSR, sparsity3, IndexType.GTE);
	}
	
	@Test
	public void testSparseBlockMCSR1LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSR, sparsity1, IndexType.LTE);
	}
	
	@Test
	public void testSparseBlockMCSR2LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSR, sparsity2, IndexType.LTE);
	}
	
	@Test
	public void testSparseBlockMCSR3LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSR, sparsity3, IndexType.LTE);
	}

	@Test
	public void testSparseBlockCSR1GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.CSR, sparsity1, IndexType.GT);
	}
	
	@Test
	public void testSparseBlockCSR2GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.CSR, sparsity2, IndexType.GT);
	}
	
	@Test
	public void testSparseBlockCSR3GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.CSR, sparsity3, IndexType.GT);
	}
	
	@Test
	public void testSparseBlockCSR1GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.CSR, sparsity1, IndexType.GTE);
	}
	
	@Test
	public void testSparseBlockCSR2GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.CSR, sparsity2, IndexType.GTE);
	}
	
	@Test
	public void testSparseBlockCSR3GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.CSR, sparsity3, IndexType.GTE);
	}
	
	@Test
	public void testSparseBlockCSR1LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.CSR, sparsity1, IndexType.LTE);
	}
	
	@Test
	public void testSparseBlockCSR2LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.CSR, sparsity2, IndexType.LTE);
	}
	
	@Test
	public void testSparseBlockCSR3LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.CSR, sparsity3, IndexType.LTE);
	}

	@Test
	public void testSparseBlockCOO1GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.COO, sparsity1, IndexType.GT);
	}
	
	@Test
	public void testSparseBlockCOO2GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.COO, sparsity2, IndexType.GT);
	}
	
	@Test
	public void testSparseBlockCOO3GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.COO, sparsity3, IndexType.GT);
	}
	
	@Test
	public void testSparseBlockCOO1GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.COO, sparsity1, IndexType.GTE);
	}
	
	@Test
	public void testSparseBlockCOO2GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.COO, sparsity2, IndexType.GTE);
	}
	
	@Test
	public void testSparseBlockCOO3GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.COO, sparsity3, IndexType.GTE);
	}
	
	@Test
	public void testSparseBlockCOO1LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.COO, sparsity1, IndexType.LTE);
	}
	
	@Test
	public void testSparseBlockCOO2LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.COO, sparsity2, IndexType.LTE);
	}
	
	@Test
	public void testSparseBlockCOO3LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.COO, sparsity3, IndexType.LTE);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runSparseBlockGetFirstIndexTest( SparseBlock.Type btype, double sparsity, IndexType itype)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 3456); 
			
			//init sparse block
			SparseBlock sblock = null;
			MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
			SparseBlock srtmp = mbtmp.getSparseBlock();			
			switch( btype ) {
				case MCSR: sblock = new SparseBlockMCSR(srtmp); break;
				case CSR: sblock = new SparseBlockCSR(srtmp); break;
				case COO: sblock = new SparseBlockCOO(srtmp); break;
			}
			
			//check for correct number of non-zeros
			int[] rnnz = new int[rows]; int nnz = 0;
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<cols; j++ )
					rnnz[i] += (A[i][j]!=0) ? 1 : 0;
				nnz += rnnz[i];
			}
			if( nnz != sblock.size() )
				Assert.fail("Wrong number of non-zeros: "+sblock.size()+", expected: "+nnz);
		
			//check correct isEmpty return
			for( int i=0; i<rows; i++ )
				if( sblock.isEmpty(i) != (rnnz[i]==0) )
					Assert.fail("Wrong isEmpty(row) result for row nnz: "+rnnz[i]);
		
			//check correct index values	
			for( int i=0; i<rows; i++ ) {
				int ix = getFirstIx(A, i, i, itype);
				int sixpos = -1;
				switch( itype ) {
					case GT: sixpos = sblock.posFIndexGT(i, i); break;
					case GTE: sixpos = sblock.posFIndexGTE(i, i); break;
					case LTE: sixpos = sblock.posFIndexLTE(i, i); break;
				}
				int six = (sixpos>=0) ? sblock.indexes(i)[sixpos] : -1;
				if( six != ix ) {
					Assert.fail("Wrong index returned by index probe ("+
							itype.toString()+","+i+"): "+six+", expected: "+ix);	
				}
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
	
	/**
	 * 
	 * @param A
	 * @param rix
	 * @param cix
	 * @param type
	 * @return
	 */
	private int getFirstIx( double[][] A, int rix, int cix, IndexType type ) {
		if( type==IndexType.GT ) {
			for( int j=cix+1; j<cols; j++ )
				if( A[rix][j] != 0 )
					return j;
			return -1;	
		}
		else if( type==IndexType.GTE ) {
			for( int j=cix; j<cols; j++ )
				if( A[rix][j] != 0 )
					return j;
			return -1;	
		}
		else if( type==IndexType.LTE ) {
			for( int j=cix; j>=0; j-- )
				if( A[rix][j] != 0 )
					return j;
			return -1;	
		}
		
		return -1;
	}
}
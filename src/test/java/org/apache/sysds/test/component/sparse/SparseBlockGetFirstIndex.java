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

import org.apache.sysds.runtime.data.SparseBlockDCSR;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCOO;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSC;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

/**
 * This is a sparse matrix block component test for sparse block get
 * first index functionality. In order to achieve broad coverage, we 
 * test against GT, GTE, and LTE as well as different sparsity values.
 * 
 */
public class SparseBlockGetFirstIndex extends AutomatedTestBase 
{
	private final static int rows = 595;
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

	@Test
	public void testSparseBlockDCSR1GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.DCSR, sparsity1, IndexType.GT);
	}

	@Test
	public void testSparseBlockDCSR2GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.DCSR, sparsity2, IndexType.GT);
	}

	@Test
	public void testSparseBlockDCSR3GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.DCSR, sparsity3, IndexType.GT);
	}

	@Test
	public void testSparseBlockDCSR1GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.DCSR, sparsity1, IndexType.GTE);
	}

	@Test
	public void testSparseBlockDCSR2GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.DCSR, sparsity2, IndexType.GTE);
	}

	@Test
	public void testSparseBlockDCSR3GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.DCSR, sparsity3, IndexType.GTE);
	}

	@Test
	public void testSparseBlockDCSR1LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.DCSR, sparsity1, IndexType.LTE);
	}

	@Test
	public void testSparseBlockDCSR2LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.DCSR, sparsity2, IndexType.LTE);
	}

	@Test
	public void testSparseBlockDCSR3LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.DCSR, sparsity3, IndexType.LTE);
	}

	@Test
	public void testSparseBlockMCSC1GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSC, sparsity1, IndexType.GT);
	}

	@Test
	public void testSparseBlockMCSC2GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSC, sparsity2, IndexType.GT);
	}

	@Test
	public void testSparseBlockMCSC3GT()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSC, sparsity3, IndexType.GT);
	}

	@Test
	public void testSparseBlockMCSC1GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSC, sparsity1, IndexType.GTE);
	}

	@Test
	public void testSparseBlockMCSC2GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSC, sparsity2, IndexType.GTE);
	}

	@Test
	public void testSparseBlockMCSC3GTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSC, sparsity3, IndexType.GTE);
	}

	@Test
	public void testSparseBlockMCSC1LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSC, sparsity1, IndexType.LTE);
	}

	@Test
	public void testSparseBlockMCSC2LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSC, sparsity2, IndexType.LTE);
	}

	@Test
	public void testSparseBlockMCSC3LTE()  {
		runSparseBlockGetFirstIndexTest(SparseBlock.Type.MCSC, sparsity3, IndexType.LTE);
	}
	
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
				case DCSR: sblock = new SparseBlockDCSR(srtmp); break;
				case MCSC: sblock = new SparseBlockMCSC(srtmp, cols); break;
			}
			
			//check for correct number of non-zeros
			int[] rnnz = new int[rows]; int nnz = 0;
			int[] cnnz =new int[cols];
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<cols; j++ ) {
					cnnz[j] += (A[i][j] != 0) ? 1 : 0;
					rnnz[i] += (A[i][j] != 0) ? 1 : 0;
				}
				nnz += rnnz[i];
			}
			if( nnz != sblock.size() )
				Assert.fail("Wrong number of non-zeros: "+sblock.size()+", expected: "+nnz);

			//check correct isEmpty return
			if(sblock instanceof SparseBlockMCSC) {
				for(int i = 0; i < cols; i++)
					if(sblock.isEmpty(i) != (cnnz[i] == 0))
						Assert.fail("Wrong isEmpty(col) result for row nnz: " + cnnz[i]);
			}
			else {
				for(int i = 0; i < rows; i++)
					if(sblock.isEmpty(i) != (rnnz[i] == 0))
						Assert.fail("Wrong isEmpty(row) result for row nnz: " + rnnz[i]);
			}

			//check correct index values
			if(sblock instanceof SparseBlockMCSC){
				for (int i = 0; i < cols; i++) {
					int ix = getFirstIxCol(A, i, i, itype);
					int sixpos = -1;
					switch (itype) {
						case GT: sixpos = sblock.posFIndexGT(i, i); break;
						case GTE: sixpos = sblock.posFIndexGTE(i, i); break;
						case LTE: sixpos = sblock.posFIndexLTE(i, i); break;
					}
					int six = (sixpos >= 0) ?
						sblock.indexes(i)[sblock.pos(i) + sixpos] : -1;
					if (six != ix) {
						Assert.fail("Wrong index returned by index probe (" +
							itype.toString() + "," + i + "): " + six + ", expected: " + ix);
					}
				}
			}
			else{
				for( int i=0; i<rows; i++ ) {
					int ix = getFirstIx(A, i, i, itype);
					int sixpos = -1;
					switch( itype ) {
						case GT: sixpos = sblock.posFIndexGT(i, i); break;
						case GTE: sixpos = sblock.posFIndexGTE(i, i); break;
						case LTE: sixpos = sblock.posFIndexLTE(i, i); break;
					}
					int six = (sixpos>=0) ?
						sblock.indexes(i)[sblock.pos(i)+sixpos] : -1;
					if( six != ix ) {
						Assert.fail("Wrong index returned by index probe ("+
								itype.toString()+","+i+"): "+six+", expected: "+ix);
					}
				}
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
	
	private static int getFirstIx( double[][] A, int rix, int cix, IndexType type ) {
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

	private static int getFirstIxCol(double[][] A, int rix, int cix, IndexType type) {
		if(type == IndexType.GT) {
			for(int j = rix + 1; j < rows; j++)
				if(A[j][cix] != 0)
					return j;
			return -1;
		}
		else if(type == IndexType.GTE) {
			for(int j = rix; j < rows; j++)
				if(A[j][cix] != 0)
					return j;
			return -1;
		}
		else if(type == IndexType.LTE) {
			for(int j = rix; j >= 0; j--)
				if(A[j][cix] != 0)
					return j;
			return -1;
		}

		return -1;
	}

}

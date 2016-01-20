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
import org.apache.sysml.runtime.util.LongLongDoubleHashMap;
import org.apache.sysml.runtime.util.LongLongDoubleHashMap.LLDoubleEntry;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

/**
 * This is a sparse matrix block component test for init, get, set, 
 * and append functionality. In order to achieve broad coverage, we
 * test against different init methods and sparsity values.
 * 
 */
public class SparseBlockGetSet extends AutomatedTestBase 
{
	private final static int rows = 732;
	private final static int cols = 354;	
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.2;
	private final static double sparsity3 = 0.3;
	
	private enum InitType {
		BULK,
		SEQ_SET,
		RAND_SET,
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockMCSR1Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSR, sparsity1, InitType.BULK);
	}
	
	@Test
	public void testSparseBlockMCSR2Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSR, sparsity2, InitType.BULK);
	}
	
	@Test
	public void testSparseBlockMCSR3Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSR, sparsity3, InitType.BULK);
	}
	
	@Test
	public void testSparseBlockMCSR1Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSR, sparsity1, InitType.SEQ_SET);
	}
	
	@Test
	public void testSparseBlockMCSR2Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSR, sparsity2, InitType.SEQ_SET);
	}
	
	@Test
	public void testSparseBlockMCSR3Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSR, sparsity3, InitType.SEQ_SET);
	}
	
	@Test
	public void testSparseBlockMCSR1Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSR, sparsity1, InitType.RAND_SET);
	}
	
	@Test
	public void testSparseBlockMCSR2Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSR, sparsity2, InitType.RAND_SET);
	}
	
	@Test
	public void testSparseBlockMCSR3Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSR, sparsity3, InitType.RAND_SET);
	}
	
	@Test
	public void testSparseBlockCSR1Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.CSR, sparsity1, InitType.BULK);
	}
	
	@Test
	public void testSparseBlockCSR2Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.CSR, sparsity2, InitType.BULK);
	}
	
	@Test
	public void testSparseBlockCSR3Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.CSR, sparsity3, InitType.BULK);
	}
	
	@Test
	public void testSparseBlockCSR1Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.CSR, sparsity1, InitType.SEQ_SET);
	}
	
	@Test
	public void testSparseBlockCSR2Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.CSR, sparsity2, InitType.SEQ_SET);
	}
	
	@Test
	public void testSparseBlockCSR3Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.CSR, sparsity3, InitType.SEQ_SET);
	}
	
	@Test
	public void testSparseBlockCSR1Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.CSR, sparsity1, InitType.RAND_SET);
	}
	
	@Test
	public void testSparseBlockCSR2Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.CSR, sparsity2, InitType.RAND_SET);
	}
	
	@Test
	public void testSparseBlockCSR3Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.CSR, sparsity3, InitType.RAND_SET);
	}
	
	@Test
	public void testSparseBlockCOO1Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.COO, sparsity1, InitType.BULK);
	}
	
	@Test
	public void testSparseBlockCOO2Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.COO, sparsity2, InitType.BULK);
	}
	
	@Test
	public void testSparseBlockCOO3Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.COO, sparsity3, InitType.BULK);
	}
	
	@Test
	public void testSparseBlockCOO1Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.COO, sparsity1, InitType.SEQ_SET);
	}
	
	@Test
	public void testSparseBlockCOO2Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.COO, sparsity2, InitType.SEQ_SET);
	}
	
	@Test
	public void testSparseBlockCOO3Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.COO, sparsity3, InitType.SEQ_SET);
	}
	
	@Test
	public void testSparseBlockCOO1Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.COO, sparsity1, InitType.RAND_SET);
	}
	
	@Test
	public void testSparseBlockCOO2Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.COO, sparsity2, InitType.RAND_SET);
	}
	
	@Test
	public void testSparseBlockCOO3Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.COO, sparsity3, InitType.RAND_SET);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runSparseBlockGetSetTest( SparseBlock.Type btype, double sparsity, InitType itype)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 7654321); 
			
			//init sparse block
			SparseBlock sblock = null;
			if( itype == InitType.BULK ) {
				MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
				SparseBlock srtmp = mbtmp.getSparseBlock();			
				switch( btype ) {
					case MCSR: sblock = new SparseBlockMCSR(srtmp); break;
					case CSR: sblock = new SparseBlockCSR(srtmp); break;
					case COO: sblock = new SparseBlockCOO(srtmp); break;
				}
			}
			else if( itype == InitType.SEQ_SET || itype == InitType.RAND_SET ) {
				switch( btype ) {
					case MCSR: sblock = new SparseBlockMCSR(rows, cols); break;
					case CSR: sblock = new SparseBlockCSR(rows, cols); break;
					case COO: sblock = new SparseBlockCOO(rows, cols); break;
				}
				
				if(itype == InitType.SEQ_SET) {
					for( int i=0; i<rows; i++ )
						for( int j=0; j<cols; j++ )
							sblock.append(i, j, A[i][j]);
				}
				else if( itype == InitType.RAND_SET ) {
					LongLongDoubleHashMap map = new LongLongDoubleHashMap();
					for( int i=0; i<rows; i++ )
						for( int j=0; j<cols; j++ )
							map.addValue(i, j, A[i][j]);
					for( LLDoubleEntry e : map.extractValues() ) //random hash order
						sblock.set((int)e.key1, (int)e.key2, e.value);
				}	
			}
			
			//check basic meta data
			if( sblock.numRows() != rows )
				Assert.fail("Wrong number of rows: "+sblock.numRows()+", expected: "+rows);
			
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
		
			//check correct values			
			for( int i=0; i<rows; i++ ) 
				if( !sblock.isEmpty(i) )	
					for( int j=0; j<cols; j++ )	{
						double tmp = sblock.get(i, j);
						if( tmp != A[i][j] )
							Assert.fail("Wrong get value for cell ("+i+","+j+"): "+tmp+", expected: "+A[i][j]);
					}		
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
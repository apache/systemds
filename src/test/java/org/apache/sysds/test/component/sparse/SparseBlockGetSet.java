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
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCOO;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSC;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.LongLongDoubleHashMap;
import org.apache.sysds.runtime.util.LongLongDoubleHashMap.ADoubleEntry;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

import java.util.Iterator;

/**
 * This is a sparse matrix block component test for init, get, set, 
 * and append functionality. In order to achieve broad coverage, we
 * test against different init methods and sparsity values.
 * 
 */
public class SparseBlockGetSet extends AutomatedTestBase 
{
	private final static int rows = 132;
	private final static int cols = 60;
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

	@Test
	public void testSparseBlockDCSR1Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.DCSR, sparsity1, InitType.BULK);
	}

	@Test
	public void testSparseBlockDCSR2Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.DCSR, sparsity2, InitType.BULK);
	}

	@Test
	public void testSparseBlockDCSR3Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.DCSR, sparsity3, InitType.BULK);
	}

	@Test
	public void testSparseBlockDCSR1Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.DCSR, sparsity1, InitType.SEQ_SET);
	}

	@Test
	public void testSparseBlockDCSR2Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.DCSR, sparsity2, InitType.SEQ_SET);
	}

	@Test
	public void testSparseBlockDCSR3Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.DCSR, sparsity3, InitType.SEQ_SET);
	}

	@Test
	public void testSparseBlockDCSR1Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.DCSR, sparsity1, InitType.RAND_SET);
	}

	@Test
	public void testSparseBlockDCSR2Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.DCSR, sparsity2, InitType.RAND_SET);
	}

	@Test
	public void testSparseBlockDCSR3Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.DCSR, sparsity3, InitType.RAND_SET);
	}

	@Test
	public void testSparseBlockMCSC1Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSC, sparsity1, InitType.BULK);
	}

	@Test
	public void testSparseBlockMCSC2Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSC, sparsity2, InitType.BULK);
	}

	@Test
	public void testSparseBlockMCSC3Bulk()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSC, sparsity3, InitType.BULK);
	}

	@Test
	public void testSparseBlockMCSC1Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSC, sparsity1, InitType.SEQ_SET);
	}

	@Test
	public void testSparseBlockMCSC2Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSC, sparsity2, InitType.SEQ_SET);
	}

	@Test
	public void testSparseBlockMCSC3Seq()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSC, sparsity3, InitType.SEQ_SET);
	}

	@Test
	public void testSparseBlockMCSC1Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSC, sparsity1, InitType.RAND_SET);
	}

	@Test
	public void testSparseBlockMCSC2Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSC, sparsity2, InitType.RAND_SET);
	}

	@Test
	public void testSparseBlockMCSC3Rand()  {
		runSparseBlockGetSetTest(SparseBlock.Type.MCSC, sparsity3, InitType.RAND_SET);
	}
	
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
				sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);
			}
			else if( itype == InitType.SEQ_SET || itype == InitType.RAND_SET ) {
				switch( btype ) {
					case MCSR: sblock = new SparseBlockMCSR(rows, cols); break;
					case CSR: sblock = new SparseBlockCSR(rows, cols); break;
					case COO: sblock = new SparseBlockCOO(rows, cols); break;
					case DCSR: sblock = new SparseBlockDCSR(rows, cols); break;
					case MCSC: sblock = new SparseBlockMCSC(rows, cols); break;
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
					Iterator<ADoubleEntry> iter = map.getIterator();
					while( iter.hasNext() ) { //random hash order
						ADoubleEntry e = iter.next();
						int r = (int)e.getKey1();
						int c = (int)e.getKey2();
						sblock.set(r, c, e.value);
					}
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
					Assert.fail("Wrong isEmpty(row) result for row nnz: "+rnnz[i] + "(row: " + i + ")");

			//check correct values
			for( int i=0; i<rows; i++ )
				if( !sblock.isEmpty(i) )
					for( int j=0; j<cols; j++ ) {
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

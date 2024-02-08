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

import java.util.Iterator;

import org.apache.sysds.runtime.data.SparseBlockDCSR;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCOO;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

/**
 * This is a sparse matrix block component test for sparse block iterator 
 * functionality. In order to achieve broad coverage, we test against 
 * full and partial iterators as well as different sparsity values.
 * 
 */
public class SparseBlockIterator extends AutomatedTestBase 
{
	private final static int rows = 324;
	private final static int cols = 100;	
	private final static int rlPartial = 134;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.2;
	private final static double sparsity3 = 0.3;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockMCSR1Full()  {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity1, false);
	}
	
	@Test
	public void testSparseBlockMCSR2Full()  {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity2, false);
	}
	
	@Test
	public void testSparseBlockMCSR3Full()  {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity3, false);
	}
	
	@Test
	public void testSparseBlockMCSR1Partial()  {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity1, true);
	}
	
	@Test
	public void testSparseBlockMCSR2Partial()  {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity2, true);
	}
	
	@Test
	public void testSparseBlockMCSR3Partial()  {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity3, true);
	}
	
	@Test
	public void testSparseBlockCSR1Full()  {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity1, false);
	}
	
	@Test
	public void testSparseBlockCSR2Full()  {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity2, false);
	}
	
	@Test
	public void testSparseBlockCSR3Full()  {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity3, false);
	}
	
	@Test
	public void testSparseBlockCSR1Partial()  {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity1, true);
	}
	
	@Test
	public void testSparseBlockCSR2Partial()  {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity2, true);
	}
	
	@Test
	public void testSparseBlockCSR3Partial()  {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity3, true);
	}
	
	@Test
	public void testSparseBlockCOO1Full()  {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity1, false);
	}
	
	@Test
	public void testSparseBlockCOO2Full()  {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity2, false);
	}
	
	@Test
	public void testSparseBlockCOO3Full()  {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity3, false);
	}
	
	@Test
	public void testSparseBlockCOO1Partial()  {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity1, true);
	}
	
	@Test
	public void testSparseBlockCOO2Partial()  {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity2, true);
	}
	
	@Test
	public void testSparseBlockCOO3Partial()  {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity3, true);
	}

	@Test
	public void testSparseBlockDCSR1Full()  {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity1, false);
	}

	@Test
	public void testSparseBlockDCSR2Full()  {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity2, false);
	}

	@Test
	public void testSparseBlockDCSR3Full()  {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity3, false);
	}

	@Test
	public void testSparseBlockDCSR1Partial()  {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity1, true);
	}

	@Test
	public void testSparseBlockDCSR2Partial()  {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity2, true);
	}

	@Test
	public void testSparseBlockDCSR3Partial()  {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity3, true);
	}
	
	private void runSparseBlockIteratorTest( SparseBlock.Type btype, double sparsity, boolean partial)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 8765432); 
			
			//init sparse block
			SparseBlock sblock = null;
			MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
			SparseBlock srtmp = mbtmp.getSparseBlock();			
			switch( btype ) {
				case MCSR: sblock = new SparseBlockMCSR(srtmp); break;
				case CSR: sblock = new SparseBlockCSR(srtmp); break;
				case COO: sblock = new SparseBlockCOO(srtmp); break;
				case DCSR: sblock = new SparseBlockDCSR(srtmp); break;
			}
			
			//check for correct number of non-zeros
			int[] rnnz = new int[rows]; int nnz = 0;
			int rl = partial ? rlPartial : 0;
			for( int i=rl; i<rows; i++ ) {
				for( int j=0; j<cols; j++ )
					rnnz[i] += (A[i][j]!=0) ? 1 : 0;
				nnz += rnnz[i];
			}
			if( !partial && nnz != sblock.size() )
				Assert.fail("Wrong number of non-zeros: "+sblock.size()+", expected: "+nnz);
		
			//check correct isEmpty return
			for( int i=rl; i<rows; i++ )
				if( sblock.isEmpty(i) != (rnnz[i]==0) )
					Assert.fail("Wrong isEmpty(row) result for row nnz: "+rnnz[i]);
		
			//check correct values	
			Iterator<IJV> iter = !partial ? sblock.getIterator() :
					sblock.getIterator(rl, rows);
			int count = 0;
			while( iter.hasNext() ) {
				IJV cell = iter.next();
				if( cell.getV() != A[cell.getI()][cell.getJ()] )
					Assert.fail("Wrong value returned by iterator: "+cell.getV()+", expected: "+A[cell.getI()][cell.getJ()]);	
				count++;
			}
			if( count != nnz )
				Assert.fail("Wrong number of values returned by iterator: "+count+", expected: "+nnz);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
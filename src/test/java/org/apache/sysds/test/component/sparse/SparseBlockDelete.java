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

import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

/**
 * This is a sparse matrix block component test for sparse block delete 
 * via set functionality. In order to achieve broad coverage, we test against 
 * different sparsity values.
 * 
 */
public class SparseBlockDelete extends AutomatedTestBase 
{
	private final static int rows = 132;
	private final static int cols = 98;	
	private final static int cl = 32;
	private final static int cu = 44;
	private final static double sparsity1 = 0.12;
	private final static double sparsity2 = 0.22;
	private final static double sparsity3 = 0.32;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockMCSR1()  {
		runSparseBlockDeleteTest(SparseBlock.Type.MCSR, sparsity1);
	}
	
	@Test
	public void testSparseBlockMCSR2()  {
		runSparseBlockDeleteTest(SparseBlock.Type.MCSR, sparsity2);
	}
	
	@Test
	public void testSparseBlockMCSR3()  {
		runSparseBlockDeleteTest(SparseBlock.Type.MCSR, sparsity3);
	}
	
	@Test
	public void testSparseBlockCSR1()  {
		runSparseBlockDeleteTest(SparseBlock.Type.CSR, sparsity1);
	}
	
	@Test
	public void testSparseBlockCSR2()  {
		runSparseBlockDeleteTest(SparseBlock.Type.CSR, sparsity2);
	}
	
	@Test
	public void testSparseBlockCSR3()  {
		runSparseBlockDeleteTest(SparseBlock.Type.CSR, sparsity3);
	}
	
	@Test
	public void testSparseBlockCOO1()  {
		runSparseBlockDeleteTest(SparseBlock.Type.COO, sparsity1);
	}
	
	@Test
	public void testSparseBlockCOO2()  {
		runSparseBlockDeleteTest(SparseBlock.Type.COO, sparsity2);
	}
	
	@Test
	public void testSparseBlockCOO3()  {
		runSparseBlockDeleteTest(SparseBlock.Type.COO, sparsity3);
	}

	@Test
	public void testSparseBlockDCSR1()  {
		runSparseBlockDeleteTest(SparseBlock.Type.DCSR, sparsity1);
	}

	@Test
	public void testSparseBlockDCSR2()  {
		runSparseBlockDeleteTest(SparseBlock.Type.DCSR, sparsity2);
	}

	@Test
	public void testSparseBlockDCSR3()  {
		runSparseBlockDeleteTest(SparseBlock.Type.DCSR, sparsity3);
	}

	@Test
	public void testSparseBlockMCSC1()  {
		runSparseBlockDeleteTest(SparseBlock.Type.MCSC, sparsity1);
	}

	@Test
	public void testSparseBlockMCSC2()  {
		runSparseBlockDeleteTest(SparseBlock.Type.MCSC, sparsity2);
	}

	@Test
	public void testSparseBlockMCSC3()  {
		runSparseBlockDeleteTest(SparseBlock.Type.MCSC, sparsity3);
	}
	
	private void runSparseBlockDeleteTest( SparseBlock.Type btype, double sparsity)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 456); 
			
			//init sparse block
			MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
			SparseBlock srtmp = mbtmp.getSparseBlock();			
			SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true, cols);
			
			//delete range per row via set
			for( int i=0; i<rows; i++ )
				for( int j=cl; j<cu; j++ ) {
					A[i][j] = 0;
					sblock.set(i, j, 0);
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
		
			//check correct values	
			Iterator<IJV> iter = sblock.getIterator();
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

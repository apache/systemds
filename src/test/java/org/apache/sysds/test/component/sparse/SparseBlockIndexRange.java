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

import java.util.Arrays;
import java.util.Iterator;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCOO;
import org.apache.sysds.runtime.data.SparseBlockCSC;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockDCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSC;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

/**
 * This is a sparse matrix block component test for sparse block deleteIndexRange
 * and setIndexRange functionality. In order to achieve broad coverage, we test 
 * against different update types and sparsity values.
 * 
 */
public class SparseBlockIndexRange extends AutomatedTestBase 
{
	private final static int rows = 662;
	private final static int cols = 549;	
	private final static int cl = 245;
	private final static int cu = 425;
	private final static double sparsity1 = 0.12;
	private final static double sparsity2 = 0.22;
	private final static double sparsity3 = 0.32;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	public enum UpdateType {
		DELETE,
		INSERT,
	}
	
	@Test
	public void testSparseBlockMCSR1Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.MCSR, sparsity1, UpdateType.DELETE);
	}
	
	@Test
	public void testSparseBlockMCSR2Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.MCSR, sparsity2, UpdateType.DELETE);
	}
	
	@Test
	public void testSparseBlockMCSR3Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.MCSR, sparsity3, UpdateType.DELETE);
	}
	
	@Test
	public void testSparseBlockMCSR1Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.MCSR, sparsity1, UpdateType.INSERT);
	}
	
	@Test
	public void testSparseBlockMCSR2Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.MCSR, sparsity2, UpdateType.INSERT);
	}
	
	@Test
	public void testSparseBlockMCSR3Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.MCSR, sparsity3, UpdateType.INSERT);
	}
	
	@Test
	public void testSparseBlockCSR1Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.CSR, sparsity1, UpdateType.DELETE);
	}
	
	@Test
	public void testSparseBlockCSR2Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.CSR, sparsity2, UpdateType.DELETE);
	}
	
	@Test
	public void testSparseBlockCSR3Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.CSR, sparsity3, UpdateType.DELETE);
	}
	
	@Test
	public void testSparseBlockCSR1Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.CSR, sparsity1, UpdateType.INSERT);
	}
	
	@Test
	public void testSparseBlockCSR2Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.CSR, sparsity2, UpdateType.INSERT);
	}
	
	@Test
	public void testSparseBlockCSR3Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.CSR, sparsity3, UpdateType.INSERT);
	}
	
	@Test
	public void testSparseBlockCOO1Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.COO, sparsity1, UpdateType.DELETE);
	}
	
	@Test
	public void testSparseBlockCOO2Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.COO, sparsity2, UpdateType.DELETE);
	}
	
	@Test
	public void testSparseBlockCOO3Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.COO, sparsity3, UpdateType.DELETE);
	}
	
	@Test
	public void testSparseBlockCOO1Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.COO, sparsity1, UpdateType.INSERT);
	}
	
	@Test
	public void testSparseBlockCOO2Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.COO, sparsity2, UpdateType.INSERT);
	}
	
	@Test
	public void testSparseBlockCOO3Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.COO, sparsity3, UpdateType.INSERT);
	}

	@Test
	public void testSparseBlockDCSR1Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.DCSR, sparsity1, UpdateType.DELETE);
	}

	@Test
	public void testSparseBlockDCSR2Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.DCSR, sparsity2, UpdateType.DELETE);
	}

	@Test
	public void testSparseBlockDCSR3Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.DCSR, sparsity3, UpdateType.DELETE);
	}

	@Test
	public void testSparseBlockDCSR1Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.DCSR, sparsity1, UpdateType.INSERT);
	}

	@Test
	public void testSparseBlockDCSR2Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.DCSR, sparsity2, UpdateType.INSERT);
	}

	@Test
	public void testSparseBlockDCSR3Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.DCSR, sparsity3, UpdateType.INSERT);
	}

	@Test
	public void testSparseBlockMCSC1Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.MCSC, sparsity1, UpdateType.DELETE);
	}

	@Test
	public void testSparseBlockMCSC2Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.MCSC, sparsity2, UpdateType.DELETE);
	}

	@Test
	public void testSparseBlockMCSC3Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.MCSC, sparsity3, UpdateType.DELETE);
	}

	@Test
	public void testSparseBlockMCSC1Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.MCSC, sparsity1, UpdateType.INSERT);
	}

	@Test
	public void testSparseBlockMCSC2Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.MCSC, sparsity2, UpdateType.INSERT);
	}

	@Test
	public void testSparseBlockMCSC3Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.MCSC, sparsity3, UpdateType.INSERT);
	}

	@Test
	public void testSparseBlockCSC1Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.CSC, sparsity1, UpdateType.DELETE);
	}

	@Test
	public void testSparseBlockCSC2Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.CSC, sparsity2, UpdateType.DELETE);
	}

	@Test
	public void testSparseBlockCSC3Delete()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.CSC, sparsity3, UpdateType.DELETE);
	}

	@Test
	public void testSparseBlockCSC1Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.CSC, sparsity1, UpdateType.INSERT);
	}

	@Test
	public void testSparseBlockCSC2Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.CSC, sparsity2, UpdateType.INSERT);
	}

	@Test
	public void testSparseBlockCSC3Insert()  {
		runSparseBlockIndexRangeTest(SparseBlock.Type.CSC, sparsity3, UpdateType.INSERT);
	}
	
	private void runSparseBlockIndexRangeTest( SparseBlock.Type btype, double sparsity, UpdateType utype)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 456); 
			
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
				case CSC: sblock = new SparseBlockCSC(srtmp, cols); break;
			}
			
			//delete range per row via set
			if( utype == UpdateType.DELETE ) {
				for( int i=0; i<rows; i++ ) {
					sblock.deleteIndexRange(i, cl, cu);
					Arrays.fill(A[i], cl, cu, 0);
				}
			}
			else if( utype == UpdateType.INSERT ) {
				double[] vals = new double[cu-cl];
				for( int j=cl; j<cu; j++ )
					vals[j-cl] = j;
				for( int i=0; i<rows; i++ ) {
					sblock.setIndexRange(i, cl, cu, vals, 0, cu-cl);
					System.arraycopy(vals, 0, A[i], cl, cu-cl);
				}
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

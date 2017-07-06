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

package org.apache.sysml.test.integration.functions.compress;

import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

/**
 * 
 */
public class BasicMatrixQuantileTest extends AutomatedTestBase
{	
	private static final int rows = 2071;
	private static final int cols = 1;	
	private static final double sparsity1 = 0.9;
	private static final double sparsity2 = 0.1;
	private static final double sparsity3 = 0.0;
	
	public enum SparsityType {
		DENSE,
		SPARSE,
		EMPTY,
	}
	
	public enum ValueType {
		RAND, //UC
		CONST, //RLE
		RAND_ROUND_OLE, //OLE
		RAND_ROUND_DDC, //RLE
	}
	
	@Override
	public void setUp() {
		
	}
	
	@Test
	public void testDenseRandDataCompression() {
		runMatrixAppendTest(SparsityType.DENSE, ValueType.RAND, true);
	}
	
	@Test
	public void testSparseRandDataCompression() {
		runMatrixAppendTest(SparsityType.SPARSE, ValueType.RAND, true);
	}
	
	@Test
	public void testEmptyCompression() {
		runMatrixAppendTest(SparsityType.EMPTY, ValueType.RAND, true);
	}
	
	@Test
	public void testDenseRoundRandDataOLECompression() {
		runMatrixAppendTest(SparsityType.DENSE, ValueType.RAND_ROUND_OLE, true);
	}
	
	@Test
	public void testSparseRoundRandDataOLECompression() {
		runMatrixAppendTest(SparsityType.SPARSE, ValueType.RAND_ROUND_OLE, true);
	}
	
	@Test
	public void testDenseRoundRandDataDDCCompression() {
		runMatrixAppendTest(SparsityType.DENSE, ValueType.RAND_ROUND_DDC, true);
	}
	
	@Test
	public void testSparseRoundRandDataDDCCompression() {
		runMatrixAppendTest(SparsityType.SPARSE, ValueType.RAND_ROUND_DDC, true);
	}
	
	@Test
	public void testDenseConstantDataCompression() {
		runMatrixAppendTest(SparsityType.DENSE, ValueType.CONST, true);
	}
	
	@Test
	public void testSparseConstDataCompression() {
		runMatrixAppendTest(SparsityType.SPARSE, ValueType.CONST, true);
	}
	
	@Test
	public void testDenseRandDataNoCompression() {
		runMatrixAppendTest(SparsityType.DENSE, ValueType.RAND, false);
	}
	
	@Test
	public void testSparseRandDataNoCompression() {
		runMatrixAppendTest(SparsityType.SPARSE, ValueType.RAND, false);
	}
	
	@Test
	public void testEmptyNoCompression() {
		runMatrixAppendTest(SparsityType.EMPTY, ValueType.RAND, false);
	}
	
	@Test
	public void testDenseRoundRandDataOLENoCompression() {
		runMatrixAppendTest(SparsityType.DENSE, ValueType.RAND_ROUND_OLE, false);
	}
	
	@Test
	public void testSparseRoundRandDataOLENoCompression() {
		runMatrixAppendTest(SparsityType.SPARSE, ValueType.RAND_ROUND_OLE, false);
	}
	
	@Test
	public void testDenseConstDataNoCompression() {
		runMatrixAppendTest(SparsityType.DENSE, ValueType.CONST, false);
	}
	
	@Test
	public void testSparseConstDataNoCompression() {
		runMatrixAppendTest(SparsityType.SPARSE, ValueType.CONST, false);
	}
	

	/**
	 * 
	 * @param mb
	 */
	private void runMatrixAppendTest(SparsityType sptype, ValueType vtype, boolean compress)
	{
		try
		{
			//prepare sparsity for input data
			double sparsity = -1;
			switch( sptype ){
				case DENSE: sparsity = sparsity1; break;
				case SPARSE: sparsity = sparsity2; break;
				case EMPTY: sparsity = sparsity3; break;
			}
			
			//generate input data
			double min = (vtype==ValueType.CONST)? 10 : -10;
			double[][] input = TestUtils.generateTestMatrix(rows, cols, min, 10, sparsity, 7);
			if( vtype==ValueType.RAND_ROUND_OLE || vtype==ValueType.RAND_ROUND_DDC ) {
				CompressedMatrixBlock.ALLOW_DDC_ENCODING = (vtype==ValueType.RAND_ROUND_DDC);
				input = TestUtils.round(input);
			}
			MatrixBlock mb = DataConverter.convertToMatrixBlock(input);
			
			//compress given matrix block
			CompressedMatrixBlock cmb = new CompressedMatrixBlock(mb);
			if( compress )
				cmb.compress();
			
			//quantile uncompressed
			MatrixBlock tmp1 = (MatrixBlock) mb.sortOperations(null, new MatrixBlock());
			double ret1 = tmp1.pickValue(0.95);
			
			//quantile compressed
			MatrixBlock tmp2 = (MatrixBlock) cmb.sortOperations(null, new MatrixBlock());
			double ret2 = tmp2.pickValue(0.95);
			
			//compare result with input
			TestUtils.compareScalars(ret1, ret2, 0.0000001);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			CompressedMatrixBlock.ALLOW_DDC_ENCODING = true;
		}
	}
}

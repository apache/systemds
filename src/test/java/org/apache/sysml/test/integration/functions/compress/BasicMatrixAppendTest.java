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
public class BasicMatrixAppendTest extends AutomatedTestBase
{	
	private static final int rows = 2071;
	private static final int cols1 = 10;
	private static final int cols2 = 1;	
	private static final double sparsity1 = 0.9;
	private static final double sparsity2 = 0.1;
	private static final double sparsity3 = 0.0;
	
	public enum SparsityType {
		DENSE,
		SPARSE,
		EMPTY,
	}
	
	public enum ValueType {
		RAND,
		RAND_ROUND,
		CONST,
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
	public void testDenseRoundRandDataCompression() {
		runMatrixAppendTest(SparsityType.DENSE, ValueType.RAND_ROUND, true);
	}
	
	@Test
	public void testSparseRoundRandDataCompression() {
		runMatrixAppendTest(SparsityType.SPARSE, ValueType.RAND_ROUND, true);
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
	public void testDenseRoundRandDataNoCompression() {
		runMatrixAppendTest(SparsityType.DENSE, ValueType.RAND_ROUND, false);
	}
	
	@Test
	public void testSparseRoundRandDataNoCompression() {
		runMatrixAppendTest(SparsityType.SPARSE, ValueType.RAND_ROUND, false);
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
			double[][] input = TestUtils.generateTestMatrix(rows, cols1, min, 10, sparsity, 7);
			if( vtype==ValueType.RAND_ROUND )
				input = TestUtils.round(input);
			MatrixBlock mb = DataConverter.convertToMatrixBlock(input);
			MatrixBlock vector = DataConverter.convertToMatrixBlock(
					TestUtils.generateTestMatrix(rows, cols2, 1, 1, 1.0, 3));
			
			//compress given matrix block
			CompressedMatrixBlock cmb = new CompressedMatrixBlock(mb);
			if( compress )
				cmb.compress();
			
			//matrix-vector uncompressed
			MatrixBlock ret1 = (MatrixBlock)mb.appendOperations(vector, new MatrixBlock());
			
			//matrix-vector compressed
			MatrixBlock ret2 = cmb.appendOperations(vector, new MatrixBlock());
			if( compress )
				ret2 = ((CompressedMatrixBlock)ret2).decompress();
			
			//compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			TestUtils.compareMatrices(d1, d2, rows, cols1+cols2, 0);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}
}

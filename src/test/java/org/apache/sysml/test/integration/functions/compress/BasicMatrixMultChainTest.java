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

import org.apache.sysml.lops.MapMultChain.ChainType;
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

/**
 * 
 */
public class BasicMatrixMultChainTest extends AutomatedTestBase
{	
	private static final int rows = 2701;
	private static final int cols = 14;
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
	public void testDenseRandDataNoWeightsCompression() {
		runMatrixMultChainTest(SparsityType.DENSE, ValueType.RAND, ChainType.XtXv, true);
	}
	
	@Test
	public void testSparseRandDataNoWeightsCompression() {
		runMatrixMultChainTest(SparsityType.SPARSE, ValueType.RAND, ChainType.XtXv, true);
	}
	
	@Test
	public void testEmptyNoWeightsCompression() {
		runMatrixMultChainTest(SparsityType.EMPTY, ValueType.RAND, ChainType.XtXv, true);
	}
	
	@Test
	public void testDenseRoundRandDataNoWeightsCompression() {
		runMatrixMultChainTest(SparsityType.DENSE, ValueType.RAND_ROUND, ChainType.XtXv, true);
	}
	
	@Test
	public void testSparseRoundRandDataNoWeightsCompression() {
		runMatrixMultChainTest(SparsityType.SPARSE, ValueType.RAND_ROUND, ChainType.XtXv, true);
	}
	
	@Test
	public void testDenseConstantDataNoWeightsCompression() {
		runMatrixMultChainTest(SparsityType.DENSE, ValueType.CONST, ChainType.XtXv, true);
	}
	
	@Test
	public void testSparseConstDataNoWeightsCompression() {
		runMatrixMultChainTest(SparsityType.SPARSE, ValueType.CONST, ChainType.XtXv, true);
	}
	
	@Test
	public void testDenseRandDataNoWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.DENSE, ValueType.RAND, ChainType.XtXv, false);
	}
	
	@Test
	public void testSparseRandDataNoWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.SPARSE, ValueType.RAND, ChainType.XtXv, false);
	}
	
	@Test
	public void testEmptyNoWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.EMPTY, ValueType.RAND, ChainType.XtXv, false);
	}
	
	@Test
	public void testDenseRoundRandDataNoWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.DENSE, ValueType.RAND_ROUND, ChainType.XtXv, false);
	}
	
	@Test
	public void testSparseRoundRandDataNoWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.SPARSE, ValueType.RAND_ROUND, ChainType.XtXv, false);
	}
	
	@Test
	public void testDenseConstDataNoWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.DENSE, ValueType.CONST, ChainType.XtXv, false);
	}
	
	@Test
	public void testSparseConstDataNoWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.SPARSE, ValueType.CONST, ChainType.XtXv, false);
	}
	
	@Test
	public void testDenseRandDataWeightsCompression() {
		runMatrixMultChainTest(SparsityType.DENSE, ValueType.RAND, ChainType.XtwXv, true);
	}
	
	@Test
	public void testSparseRandDataWeightsCompression() {
		runMatrixMultChainTest(SparsityType.SPARSE, ValueType.RAND, ChainType.XtwXv, true);
	}
	
	@Test
	public void testEmptyWeightsCompression() {
		runMatrixMultChainTest(SparsityType.EMPTY, ValueType.RAND, ChainType.XtwXv, true);
	}
	
	@Test
	public void testDenseRoundRandDataWeightsCompression() {
		runMatrixMultChainTest(SparsityType.DENSE, ValueType.RAND_ROUND, ChainType.XtwXv, true);
	}
	
	@Test
	public void testSparseRoundRandDataWeightsCompression() {
		runMatrixMultChainTest(SparsityType.SPARSE, ValueType.RAND_ROUND, ChainType.XtwXv, true);
	}
	
	@Test
	public void testDenseConstantDataWeightsCompression() {
		runMatrixMultChainTest(SparsityType.DENSE, ValueType.CONST, ChainType.XtwXv, true);
	}
	
	@Test
	public void testSparseConstDataWeightsCompression() {
		runMatrixMultChainTest(SparsityType.SPARSE, ValueType.CONST, ChainType.XtwXv, true);
	}
	
	@Test
	public void testDenseRandDataWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.DENSE, ValueType.RAND, ChainType.XtwXv, false);
	}
	
	@Test
	public void testSparseRandDataWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.SPARSE, ValueType.RAND, ChainType.XtwXv, false);
	}
	
	@Test
	public void testEmptyWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.EMPTY, ValueType.RAND, ChainType.XtwXv, false);
	}
	
	@Test
	public void testDenseRoundRandDataWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.DENSE, ValueType.RAND_ROUND, ChainType.XtwXv, false);
	}
	
	@Test
	public void testSparseRoundRandDataWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.SPARSE, ValueType.RAND_ROUND, ChainType.XtwXv, false);
	}
	
	@Test
	public void testDenseConstDataWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.DENSE, ValueType.CONST, ChainType.XtwXv, false);
	}
	
	@Test
	public void testSparseConstDataWeightsNoCompression() {
		runMatrixMultChainTest(SparsityType.SPARSE, ValueType.CONST, ChainType.XtwXv, false);
	}

	/**
	 * 
	 * @param mb
	 */
	private void runMatrixMultChainTest(SparsityType sptype, ValueType vtype, ChainType ctype, boolean compress)
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
			if( vtype==ValueType.RAND_ROUND )
				input = TestUtils.round(input);
			MatrixBlock mb = DataConverter.convertToMatrixBlock(input);
			MatrixBlock vector1 = DataConverter.convertToMatrixBlock(
					TestUtils.generateTestMatrix(cols, 1, 0, 1, 1.0, 3));
			MatrixBlock vector2 = (ctype==ChainType.XtwXv)? DataConverter.convertToMatrixBlock(
					TestUtils.generateTestMatrix(rows, 1, 0, 1, 1.0, 3)) : null;
			
			//compress given matrix block
			CompressedMatrixBlock cmb = new CompressedMatrixBlock(mb);
			if( compress )
				cmb.compress();
			
			//matrix-vector uncompressed
			MatrixBlock ret1 = (MatrixBlock)mb.chainMatrixMultOperations(vector1, vector2, new MatrixBlock(), ctype);
			
			//matrix-vector compressed
			MatrixBlock ret2 = (MatrixBlock)cmb.chainMatrixMultOperations(vector1, vector2, new MatrixBlock(), ctype);
			
			//compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			TestUtils.compareMatrices(d1, d2, cols, 1, 0.0000001);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}
}

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

import org.apache.sysml.runtime.compress.BitmapEncoder;
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

/**
 * 
 */
public class ParCompressionTest extends AutomatedTestBase
{	
	//large enough for multiple parallel tasks and mis-aligned
	private static final int rows = 3 * BitmapEncoder.BITMAP_BLOCK_SZ + 7;
	private static final int cols = 20;
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
		runCompressionTest(SparsityType.DENSE, ValueType.RAND, true);
	}
	
	@Test
	public void testSparseRandDataCompression() {
		runCompressionTest(SparsityType.SPARSE, ValueType.RAND, true);
	}
	
	@Test
	public void testEmptyCompression() {
		runCompressionTest(SparsityType.EMPTY, ValueType.RAND, true);
	}
	
	@Test
	public void testDenseRoundRandDataCompression() {
		runCompressionTest(SparsityType.DENSE, ValueType.RAND_ROUND, true);
	}
	
	@Test
	public void testSparseRoundRandDataCompression() {
		runCompressionTest(SparsityType.SPARSE, ValueType.RAND_ROUND, true);
	}
	
	@Test
	public void testDenseConstantDataCompression() {
		runCompressionTest(SparsityType.DENSE, ValueType.CONST, true);
	}
	
	@Test
	public void testSparseConstDataCompression() {
		runCompressionTest(SparsityType.SPARSE, ValueType.CONST, true);
	}
	
	@Test
	public void testDenseRandDataNoCompression() {
		runCompressionTest(SparsityType.DENSE, ValueType.RAND, false);
	}
	
	@Test
	public void testSparseRandDataNoCompression() {
		runCompressionTest(SparsityType.SPARSE, ValueType.RAND, false);
	}
	
	@Test
	public void testEmptyNoCompression() {
		runCompressionTest(SparsityType.EMPTY, ValueType.RAND, false);
	}
	
	@Test
	public void testDenseRoundRandDataNoCompression() {
		runCompressionTest(SparsityType.DENSE, ValueType.RAND_ROUND, false);
	}
	
	@Test
	public void testSparseRoundRandDataNoCompression() {
		runCompressionTest(SparsityType.SPARSE, ValueType.RAND_ROUND, false);
	}
	
	@Test
	public void testDenseConstDataNoCompression() {
		runCompressionTest(SparsityType.DENSE, ValueType.CONST, false);
	}
	
	@Test
	public void testSparseConstDataNoCompression() {
		runCompressionTest(SparsityType.SPARSE, ValueType.CONST, false);
	}
	

	/**
	 * 
	 * @param mb
	 */
	private void runCompressionTest(SparsityType sptype, ValueType vtype, boolean compress)
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
			int k = InfrastructureAnalyzer.getLocalParallelism();
			double min = (vtype==ValueType.CONST)? 10 : -10;
			double[][] input = TestUtils.generateTestMatrix(rows, cols, min, 10, sparsity, 7);
			if( vtype==ValueType.RAND_ROUND )
				input = TestUtils.round(input);
			MatrixBlock mb = DataConverter.convertToMatrixBlock(input);
			
			//compress given matrix block
			CompressedMatrixBlock cmb = new CompressedMatrixBlock(mb);
			if( compress )
				cmb.compress(k);
			
			//decompress the compressed matrix block
			MatrixBlock tmp = cmb.decompress(k);
			
			//compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(mb);
			double[][] d2 = DataConverter.convertToDoubleMatrix(tmp);
			TestUtils.compareMatrices(d1, d2, rows, cols, 0);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}
}

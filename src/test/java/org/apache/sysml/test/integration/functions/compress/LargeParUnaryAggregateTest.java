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
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;


/**
 * 
 */
public class LargeParUnaryAggregateTest extends AutomatedTestBase
{	
	private static final int rows = 5*BitmapEncoder.BITMAP_BLOCK_SZ;
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
	
	public enum AggType {
		ROWSUMS,
		COLSUMS,
		SUM,
		ROWSUMSSQ,
		COLSUMSSQ,
		SUMSQ,
		ROWMAXS,
		COLMAXS,
		MAX,
		ROWMINS,
		COLMINS,
		MIN,
	}
	
	@Override
	public void setUp() {
		
	}
	
	@Test
	public void testRowSumsDenseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.ROWSUMS, true);
	}
	
	@Test
	public void testRowSumsSparseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.ROWSUMS, true);
	}
	
	@Test
	public void testRowSumsEmptyCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.ROWSUMS, true);
	}
	
	@Test
	public void testRowSumsDenseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.ROWSUMS, true);
	}
	
	@Test
	public void testRowSumsSparseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.ROWSUMS, true);
	}
	
	@Test
	public void testRowSumsDenseConstantDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.ROWSUMS, true);
	}
	
	@Test
	public void testRowSumsSparseConstDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.ROWSUMS, true);
	}
	
	@Test
	public void testRowSumsDenseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.ROWSUMS, false);
	}
	
	@Test
	public void testRowSumsSparseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.ROWSUMS, false);
	}
	
	@Test
	public void testRowSumsEmptyNoCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.ROWSUMS, false);
	}
	
	@Test
	public void testRowSumsDenseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.ROWSUMS, false);
	}
	
	@Test
	public void testRowSumsSparseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.ROWSUMS, false);
	}
	
	@Test
	public void testRowSumsDenseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.ROWSUMS, false);
	}
	
	@Test
	public void testRowSumsSparseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.ROWSUMS, false);
	}
	
	@Test
	public void testColSumsDenseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.COLSUMS, true);
	}
	
	@Test
	public void testColSumsSparseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.COLSUMS, true);
	}
	
	@Test
	public void testColSumsEmptyCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.COLSUMS, true);
	}
	
	@Test
	public void testColSumsDenseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.COLSUMS, true);
	}
	
	@Test
	public void testColSumsSparseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.COLSUMS, true);
	}
	
	@Test
	public void testColSumsDenseConstantDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.COLSUMS, true);
	}
	
	@Test
	public void testColSumsSparseConstDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.COLSUMS, true);
	}
	
	@Test
	public void testColSumsDenseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.COLSUMS, false);
	}
	
	@Test
	public void testColSumsSparseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.COLSUMS, false);
	}
	
	@Test
	public void testColSumsEmptyNoCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.COLSUMS, false);
	}
	
	@Test
	public void testColSumsDenseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.COLSUMS, false);
	}
	
	@Test
	public void testColSumsSparseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.COLSUMS, false);
	}
	
	@Test
	public void testColSumsDenseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.COLSUMS, false);
	}
	
	@Test
	public void testColSumsSparseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.COLSUMS, false);
	}

	@Test
	public void testSumDenseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.SUM, true);
	}
	
	@Test
	public void testSumSparseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.SUM, true);
	}
	
	@Test
	public void testSumEmptyCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.SUM, true);
	}
	
	@Test
	public void testSumDenseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.SUM, true);
	}
	
	@Test
	public void testSumSparseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.SUM, true);
	}
	
	@Test
	public void testSumDenseConstantDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.SUM, true);
	}
	
	@Test
	public void testSumSparseConstDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.SUM, true);
	}
	
	@Test
	public void testSumDenseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.SUM, false);
	}
	
	@Test
	public void testSumSparseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.SUM, false);
	}
	
	@Test
	public void testSumEmptyNoCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.SUM, false);
	}
	
	@Test
	public void testSumDenseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.SUM, false);
	}
	
	@Test
	public void testSumSparseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.SUM, false);
	}
	
	@Test
	public void testSumDenseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.SUM, false);
	}
	
	@Test
	public void testSumSparseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.SUM, false);
	}
	
	@Test
	public void testRowSumsSqDenseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.ROWSUMSSQ, true);
	}
	
	@Test
	public void testRowSumsSqSparseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.ROWSUMSSQ, true);
	}
	
	@Test
	public void testRowSumsSqEmptyCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.ROWSUMSSQ, true);
	}
	
	@Test
	public void testRowSumsSqDenseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.ROWSUMSSQ, true);
	}
	
	@Test
	public void testRowSumsSqSparseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.ROWSUMSSQ, true);
	}
	
	@Test
	public void testRowSumsSqDenseConstantDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.ROWSUMSSQ, true);
	}
	
	//@Test
	//public void testRowSumsSqSparseConstDataCompression() {
	//	runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.ROWSUMSSQ, true);
	//}
	
	@Test
	public void testRowSumsSqDenseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.ROWSUMSSQ, false);
	}
	
	@Test
	public void testRowSumsSqSparseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.ROWSUMSSQ, false);
	}
	
	@Test
	public void testRowSumsSqEmptyNoCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.ROWSUMSSQ, false);
	}
	
	@Test
	public void testRowSumsSqDenseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.ROWSUMSSQ, false);
	}
	
	@Test
	public void testRowSumsSqSparseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.ROWSUMSSQ, false);
	}
	
	@Test
	public void testRowSumsSqDenseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.ROWSUMSSQ, false);
	}
	
	@Test
	public void testRowSumsSqSparseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.ROWSUMSSQ, false);
	}
	
	@Test
	public void testColSumsSqDenseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.COLSUMSSQ, true);
	}
	
	@Test
	public void testColSumsSqSparseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.COLSUMSSQ, true);
	}
	
	@Test
	public void testColSumsSqEmptyCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.COLSUMSSQ, true);
	}
	
	@Test
	public void testColSumsSqDenseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.COLSUMSSQ, true);
	}
	
	@Test
	public void testColSumsSqSparseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.COLSUMSSQ, true);
	}
	
	@Test
	public void testColSumsSqDenseConstantDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.COLSUMSSQ, true);
	}
	
	@Test
	public void testColSumsSqSparseConstDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.COLSUMSSQ, true);
	}
	
	@Test
	public void testColSumsSqDenseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.COLSUMSSQ, false);
	}
	
	@Test
	public void testColSumsSqSparseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.COLSUMSSQ, false);
	}
	
	@Test
	public void testColSumsSqEmptyNoCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.COLSUMSSQ, false);
	}
	
	@Test
	public void testColSumsSqDenseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.COLSUMSSQ, false);
	}
	
	@Test
	public void testColSumsSqSparseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.COLSUMSSQ, false);
	}
	
	@Test
	public void testColSumsSqDenseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.COLSUMSSQ, false);
	}
	
	@Test
	public void testColSumsSqSparseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.COLSUMSSQ, false);
	}

	@Test
	public void testSumSqDenseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.SUMSQ, true);
	}
	
	@Test
	public void testSumSqSparseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.SUMSQ, true);
	}
	
	@Test
	public void testSumSqEmptyCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.SUMSQ, true);
	}
	
	@Test
	public void testSumSqDenseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.SUMSQ, true);
	}
	
	@Test
	public void testSumSqSparseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.SUMSQ, true);
	}
	
	@Test
	public void testSumSqDenseConstantDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.SUMSQ, true);
	}
	
	@Test
	public void testSumSqSparseConstDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.SUMSQ, true);
	}
	
	@Test
	public void testSumSqDenseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.SUMSQ, false);
	}
	
	@Test
	public void testSumSqSparseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.SUMSQ, false);
	}
	
	@Test
	public void testSumSqEmptyNoCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.SUMSQ, false);
	}
	
	@Test
	public void testSumSqDenseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.SUMSQ, false);
	}
	
	@Test
	public void testSumSqSparseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.SUMSQ, false);
	}
	
	@Test
	public void testSumSqDenseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.SUMSQ, false);
	}
	
	@Test
	public void testSumSqSparseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.SUMSQ, false);
	}
	

	@Test
	public void testRowMaxsDenseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.ROWMAXS, true);
	}
	
	@Test
	public void testRowMaxsSparseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.ROWMAXS, true);
	}
	
	@Test
	public void testRowMaxsEmptyCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.ROWMAXS, true);
	}
	
	@Test
	public void testRowMaxsDenseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.ROWMAXS, true);
	}
	
	@Test
	public void testRowMaxsSparseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.ROWMAXS, true);
	}
	
	@Test
	public void testRowMaxsDenseConstantDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.ROWMAXS, true);
	}
	
	@Test
	public void testRowMaxsSparseConstDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.ROWMAXS, true);
	}
	
	@Test
	public void testRowMaxsDenseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.ROWMAXS, false);
	}
	
	@Test
	public void testRowMaxsSparseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.ROWMAXS, false);
	}
	
	@Test
	public void testRowMaxsEmptyNoCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.ROWMAXS, false);
	}
	
	@Test
	public void testRowMaxsDenseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.ROWMAXS, false);
	}
	
	@Test
	public void testRowMaxsSparseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.ROWMAXS, false);
	}
	
	@Test
	public void testRowMaxsDenseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.ROWMAXS, false);
	}
	
	@Test
	public void testRowMaxsSparseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.ROWMAXS, false);
	}
	
	@Test
	public void testColMaxsDenseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.COLMAXS, true);
	}
	
	@Test
	public void testColMaxsSparseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.COLMAXS, true);
	}
	
	@Test
	public void testColMaxsEmptyCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.COLMAXS, true);
	}
	
	@Test
	public void testColMaxsDenseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.COLMAXS, true);
	}
	
	@Test
	public void testColMaxsSparseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.COLMAXS, true);
	}
	
	@Test
	public void testColMaxsDenseConstantDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.COLMAXS, true);
	}
	
	@Test
	public void testColMaxsSparseConstDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.COLMAXS, true);
	}
	
	@Test
	public void testColMaxsDenseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.COLMAXS, false);
	}
	
	@Test
	public void testColMaxsSparseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.COLMAXS, false);
	}
	
	@Test
	public void testColMaxsEmptyNoCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.COLMAXS, false);
	}
	
	@Test
	public void testColMaxsDenseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.COLMAXS, false);
	}
	
	@Test
	public void testColMaxsSparseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.COLMAXS, false);
	}
	
	@Test
	public void testColMaxsDenseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.COLMAXS, false);
	}
	
	@Test
	public void testColMaxsSparseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.COLMAXS, false);
	}

	@Test
	public void testMaxDenseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.MAX, true);
	}
	
	@Test
	public void testMaxSparseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.MAX, true);
	}
	
	@Test
	public void testMaxEmptyCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.MAX, true);
	}
	
	@Test
	public void testMaxDenseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.MAX, true);
	}
	
	@Test
	public void testMaxSparseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.MAX, true);
	}
	
	@Test
	public void testMaxDenseConstantDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.MAX, true);
	}
	
	@Test
	public void testMaxSparseConstDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.MAX, true);
	}
	
	@Test
	public void testMaxDenseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.MAX, false);
	}
	
	@Test
	public void testMaxSparseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.MAX, false);
	}
	
	@Test
	public void testMaxEmptyNoCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.MAX, false);
	}
	
	@Test
	public void testMaxDenseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.MAX, false);
	}
	
	@Test
	public void testMaxSparseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.MAX, false);
	}
	
	@Test
	public void testMaxDenseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.MAX, false);
	}
	
	@Test
	public void testMaxSparseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.MAX, false);
	}
	
	@Test
	public void testRowMinsDenseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.ROWMINS, true);
	}
	
	@Test
	public void testRowMinsSparseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.ROWMINS, true);
	}
	
	@Test
	public void testRowMinsEmptyCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.ROWMINS, true);
	}
	
	@Test
	public void testRowMinsDenseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.ROWMINS, true);
	}
	
	@Test
	public void testRowMinsSparseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.ROWMINS, true);
	}
	
	@Test
	public void testRowMinsDenseConstantDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.ROWMINS, true);
	}
	
	@Test
	public void testRowMinsSparseConstDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.ROWMINS, true);
	}
	
	@Test
	public void testRowMinsDenseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.ROWMINS, false);
	}
	
	@Test
	public void testRowMinsSparseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.ROWMINS, false);
	}
	
	@Test
	public void testRowMinsEmptyNoCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.ROWMINS, false);
	}
	
	@Test
	public void testRowMinsDenseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.ROWMINS, false);
	}
	
	@Test
	public void testRowMinsSparseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.ROWMINS, false);
	}
	
	@Test
	public void testRowMinsDenseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.ROWMINS, false);
	}
	
	@Test
	public void testRowMinsSparseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.ROWMINS, false);
	}
	
	@Test
	public void testColMinsDenseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.COLMINS, true);
	}
	
	@Test
	public void testColMinsSparseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.COLMINS, true);
	}
	
	@Test
	public void testColMinsEmptyCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.COLMINS, true);
	}
	
	@Test
	public void testColMinsDenseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.COLMINS, true);
	}
	
	@Test
	public void testColMinsSparseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.COLMINS, true);
	}
	
	@Test
	public void testColMinsDenseConstantDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.COLMINS, true);
	}
	
	@Test
	public void testColMinsSparseConstDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.COLMINS, true);
	}
	
	@Test
	public void testColMinsDenseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.COLMINS, false);
	}
	
	@Test
	public void testColMinsSparseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.COLMINS, false);
	}
	
	@Test
	public void testColMinsEmptyNoCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.COLMINS, false);
	}
	
	@Test
	public void testColMinsDenseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.COLMINS, false);
	}
	
	@Test
	public void testColMinsSparseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.COLMINS, false);
	}
	
	@Test
	public void testColMinsDenseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.COLMINS, false);
	}
	
	@Test
	public void testColMinsSparseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.COLMINS, false);
	}

	@Test
	public void testMinDenseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.MIN, true);
	}
	
	@Test
	public void testMinSparseRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.MIN, true);
	}
	
	@Test
	public void testMinEmptyCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.MIN, true);
	}
	
	@Test
	public void testMinDenseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.MIN, true);
	}
	
	@Test
	public void testMinSparseRoundRandDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.MIN, true);
	}
	
	@Test
	public void testMinDenseConstantDataCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.MIN, true);
	}
	
	@Test
	public void testMinSparseConstDataCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.MIN, true);
	}
	
	@Test
	public void testMinDenseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND, AggType.MIN, false);
	}
	
	@Test
	public void testMinSparseRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND, AggType.MIN, false);
	}
	
	@Test
	public void testMinEmptyNoCompression() {
		runUnaryAggregateTest(SparsityType.EMPTY, ValueType.RAND, AggType.MIN, false);
	}
	
	@Test
	public void testMinDenseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.RAND_ROUND, AggType.MIN, false);
	}
	
	@Test
	public void testMinSparseRoundRandDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.RAND_ROUND, AggType.MIN, false);
	}
	
	@Test
	public void testMinDenseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.DENSE, ValueType.CONST, AggType.MIN, false);
	}
	
	@Test
	public void testMinSparseConstDataNoCompression() {
		runUnaryAggregateTest(SparsityType.SPARSE, ValueType.CONST, AggType.MIN, false);
	}
		
	/**
	 * 
	 * @param mb
	 */
	private void runUnaryAggregateTest(SparsityType sptype, ValueType vtype, AggType aggtype, boolean compress)
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
			mb = mb.appendOperations(MatrixBlock.seqOperations(0.1, rows-0.1, 1), new MatrixBlock()); //uc group
			
			//prepare unary aggregate operator
			AggregateUnaryOperator auop = null;
			switch (aggtype) {
				case SUM: auop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+"); break;
				case ROWSUMS: auop = InstructionUtils.parseBasicAggregateUnaryOperator("uark+"); break;
				case COLSUMS: auop = InstructionUtils.parseBasicAggregateUnaryOperator("uack+"); break;
				case SUMSQ: auop = InstructionUtils.parseBasicAggregateUnaryOperator("uasqk+"); break;
				case ROWSUMSSQ: auop = InstructionUtils.parseBasicAggregateUnaryOperator("uarsqk+"); break;
				case COLSUMSSQ: auop = InstructionUtils.parseBasicAggregateUnaryOperator("uacsqk+"); break;
				case MAX: auop = InstructionUtils.parseBasicAggregateUnaryOperator("uamax"); break;
				case ROWMAXS: auop = InstructionUtils.parseBasicAggregateUnaryOperator("uarmax"); break;
				case COLMAXS: auop = InstructionUtils.parseBasicAggregateUnaryOperator("uacmax"); break;
				case MIN: auop = InstructionUtils.parseBasicAggregateUnaryOperator("uamin"); break;
				case ROWMINS: auop = InstructionUtils.parseBasicAggregateUnaryOperator("uarmin"); break;
				case COLMINS: auop = InstructionUtils.parseBasicAggregateUnaryOperator("uacmin"); break;
			}
			auop.setNumThreads(InfrastructureAnalyzer.getLocalParallelism());
			
			//compress given matrix block
			CompressedMatrixBlock cmb = new CompressedMatrixBlock(mb);
			if( compress )
				cmb.compress();
			
			//matrix-vector uncompressed						
			MatrixBlock ret1 = (MatrixBlock)mb.aggregateUnaryOperations(auop, new MatrixBlock(), 1000, 1000, null, true);
			
			//matrix-vector compressed
			MatrixBlock ret2 = (MatrixBlock)cmb.aggregateUnaryOperations(auop, new MatrixBlock(), 1000, 1000, null, true);
			
			//compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			int dim1 = (aggtype == AggType.ROWSUMS || aggtype == AggType.ROWSUMSSQ 
					|| aggtype == AggType.ROWMINS || aggtype == AggType.ROWMINS)?rows:1;
			int dim2 = (aggtype == AggType.COLSUMS || aggtype == AggType.COLSUMSSQ 
					|| aggtype == AggType.COLMAXS || aggtype == AggType.COLMINS)?cols:1;
			TestUtils.compareMatrices(d1, d2, dim1, dim2, 0.00000000001);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}
}

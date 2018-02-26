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
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

public class LargeMatrixMatrixMultTest extends AutomatedTestBase
{
	private static final int rows = 5*BitmapEncoder.BITMAP_BLOCK_SZ;
	private static final int cols = 20;
	private static final int cols2 = 3;
	private static final double sparsity1 = 0.9;
	private static final double sparsity2 = 0.1;
	private static final double sparsity3 = 0.0;
	
	public enum MultType {
		LEFT,
		RIGHT,
	}
	
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
	public void testDenseRandDataCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.DENSE, ValueType.RAND, true);
	}
	
	@Test
	public void testSparseRandDataCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.SPARSE, ValueType.RAND, true);
	}
	
	@Test
	public void testEmptyCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.EMPTY, ValueType.RAND, true);
	}
	
	@Test
	public void testDenseRoundRandDataOLECompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.DENSE, ValueType.RAND_ROUND_OLE, true);
	}
	
	@Test
	public void testSparseRoundRandDataOLECompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.SPARSE, ValueType.RAND_ROUND_OLE, true);
	}
	
	@Test
	public void testDenseRoundRandDataDDCCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.DENSE, ValueType.RAND_ROUND_DDC, true);
	}
	
	@Test
	public void testSparseRoundRandDataDDCCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.SPARSE, ValueType.RAND_ROUND_DDC, true);
	}
	
	@Test
	public void testDenseConstantDataCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.DENSE, ValueType.CONST, true);
	}
	
	@Test
	public void testSparseConstDataCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.SPARSE, ValueType.CONST, true);
	}
	
	@Test
	public void testDenseRandDataNoCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.DENSE, ValueType.RAND, false);
	}
	
	@Test
	public void testSparseRandDataNoCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.SPARSE, ValueType.RAND, false);
	}
	
	@Test
	public void testEmptyNoCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.EMPTY, ValueType.RAND, false);
	}
	
	@Test
	public void testDenseRoundRandDataOLENoCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.DENSE, ValueType.RAND_ROUND_OLE, false);
	}
	
	@Test
	public void testSparseRoundRandDataOLENoCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.SPARSE, ValueType.RAND_ROUND_OLE, false);
	}
	
	@Test
	public void testDenseConstDataNoCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.DENSE, ValueType.CONST, false);
	}
	
	@Test
	public void testSparseConstDataNoCompressionRight() {
		runMatrixVectorMultTest(MultType.RIGHT, SparsityType.SPARSE, ValueType.CONST, false);
	}
	
	@Test
	public void testDenseRandDataCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.DENSE, ValueType.RAND, true);
	}
	
	@Test
	public void testSparseRandDataCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.SPARSE, ValueType.RAND, true);
	}
	
	@Test
	public void testEmptyCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.EMPTY, ValueType.RAND, true);
	}
	
	@Test
	public void testDenseRoundRandDataOLECompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.DENSE, ValueType.RAND_ROUND_OLE, true);
	}
	
	@Test
	public void testSparseRoundRandDataOLECompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.SPARSE, ValueType.RAND_ROUND_OLE, true);
	}
	
	@Test
	public void testDenseRoundRandDataDDCCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.DENSE, ValueType.RAND_ROUND_DDC, true);
	}
	
	@Test
	public void testSparseRoundRandDataDDCCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.SPARSE, ValueType.RAND_ROUND_DDC, true);
	}
	
	@Test
	public void testDenseConstantDataCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.DENSE, ValueType.CONST, true);
	}
	
	@Test
	public void testSparseConstDataCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.SPARSE, ValueType.CONST, true);
	}
	
	@Test
	public void testDenseRandDataNoCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.DENSE, ValueType.RAND, false);
	}
	
	@Test
	public void testSparseRandDataNoCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.SPARSE, ValueType.RAND, false);
	}
	
	@Test
	public void testEmptyNoCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.EMPTY, ValueType.RAND, false);
	}
	
	@Test
	public void testDenseRoundRandDataOLENoCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.DENSE, ValueType.RAND_ROUND_OLE, false);
	}
	
	@Test
	public void testSparseRoundRandDataOLENoCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.SPARSE, ValueType.RAND_ROUND_OLE, false);
	}
	
	@Test
	public void testDenseConstDataNoCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.DENSE, ValueType.CONST, false);
	}
	
	@Test
	public void testSparseConstDataNoCompressionLeft() {
		runMatrixVectorMultTest(MultType.LEFT, SparsityType.SPARSE, ValueType.CONST, false);
	}
	
	private static void runMatrixVectorMultTest(MultType mtype, SparsityType sptype, ValueType vtype, boolean compress)
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
			
			boolean right = (mtype == MultType.RIGHT);
			MatrixBlock mb = DataConverter.convertToMatrixBlock(input);
			MatrixBlock vect = DataConverter.convertToMatrixBlock(right?
					TestUtils.generateTestMatrix(cols, cols2, -1, 1, 1.0, 3) :
					TestUtils.generateTestMatrix(cols2, rows, -1, 1, 1.0, 3));
			
			//compress given matrix block
			CompressedMatrixBlock cmb = new CompressedMatrixBlock(mb);
			if( compress )
				cmb.compress();
			
			//matrix-vector uncompressed
			AggregateOperator aop = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator abop = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), aop);
			MatrixBlock ret1 = right ?
				mb.aggregateBinaryOperations(mb, vect, new MatrixBlock(), abop) :
				vect.aggregateBinaryOperations(vect, mb, new MatrixBlock(), abop);
			
			//matrix-vector compressed
			MatrixBlock ret2 = right ?
				cmb.aggregateBinaryOperations(cmb, vect, new MatrixBlock(), abop) :
				cmb.aggregateBinaryOperations(vect, cmb, new MatrixBlock(), abop);
			
			//compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			TestUtils.compareMatrices(d1, d2,
				right?rows:cols2, right?cols2:cols, 0.0000001);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			CompressedMatrixBlock.ALLOW_DDC_ENCODING = true;
		}
	}
}

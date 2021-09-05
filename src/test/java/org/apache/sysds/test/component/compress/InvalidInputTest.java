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

package org.apache.sysds.test.component.compress;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlock.Type;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class InvalidInputTest {

	final MatrixBlock mb;
	final CompressedMatrixBlock cmb;

	public InvalidInputTest() {
		final int rows = 1000;
		final int cols = 2;
		final double min = 0;
		final double max = 100;
		final double sparsity = 0.6;
		final int seed = 2;

		double[][] input = TestUtils.round(TestUtils.generateTestMatrix(rows, cols, min, max, sparsity, seed));
		mb = DataConverter.convertToMatrixBlock(input);
		mb.recomputeNonZeros();
		mb.examSparsity();

		cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb).getLeft();
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidAggregateBinaryOperationsNoneOfTheInputsIsCallingObject() {
		int k = InfrastructureAnalyzer.getLocalParallelism();
		MatrixBlock a = new MatrixBlock(10, 5, true);
		MatrixBlock b = new MatrixBlock(5, 10, true);
		// Not the typical way of allocating a Compressed Matrix Block
		MatrixBlock c = new CompressedMatrixBlock(10, 4);
		AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(k);
		c.aggregateBinaryOperations(a, b, new MatrixBlock(), abop);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalid_call_allocateAndResetSparseBlock_01() {
		cmb.allocateAndResetSparseBlock(true, Type.CSR);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalid_call_allocateAndResetSparseBlock_02() {
		cmb.allocateAndResetSparseBlock(true, Type.MCSR);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalid_call_allocateAndResetSparseBlock_03() {
		cmb.allocateAndResetSparseBlock(false, Type.CSR);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalid_call_allocateBlockAsync() {
		cmb.allocateBlockAsync();
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalid_call_allocateDenseBlock() {
		cmb.allocateDenseBlock();
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalid_call_allocateSparseRowsBlock() {
		cmb.allocateSparseRowsBlock();
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalid_call_appendRow() {
		cmb.appendRow(1, new SparseRowVector(1, 3));
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalid_call_appendRowToSparse() {
		MatrixBlock x = new MatrixBlock(100, 100, true);
		x.allocateSparseRowsBlock();
		SparseBlock sb = x.getSparseBlock();
		cmb.appendRowToSparse(sb, null, 1, 1, 1, false);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalid_call_appendToSparse() {
		cmb.appendToSparse(null, 1, 1, false);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalid_call_appendToValue() {
		cmb.appendValue(1, 1, 0.2);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalid_call_appendToValuePlain() {
		cmb.appendValuePlain(1, 1, 0.2);
	}

	@Test(expected = DMLCompressionException.class)
	public void test_copyShallow() {
		MatrixBlock copyIntoMe = new MatrixBlock();
		copyIntoMe.copyShallow(cmb);
	}

	@Test(expected = DMLCompressionException.class)
	public void test_copyShallowCompressed() {
		CompressedMatrixBlock copyIntoMe = new CompressedMatrixBlock();
		copyIntoMe.copyShallow(mb);
	}
}

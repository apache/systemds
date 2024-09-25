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

package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.junit.Test;

public class MatrixAggKernelTest {
	private static final int MIN_PAR = 2*(int)LibMatrixAgg.PAR_NUMCELL_THRESHOLD1;
	private static final int MIN_PAR_SQRT = (int)Math.sqrt(MIN_PAR);
	private static final int K = InfrastructureAnalyzer.getLocalParallelism();
	
	@Test
	public void testDenseKahanSum() {
		testMatrixAggregation("uak+", "uark+", "uack+", 0.95);
	}
	
	@Test
	public void testSparseKahanSum() {
		testMatrixAggregation("uak+", "uark+", "uack+", 0.1);
	}
	
	@Test
	public void testDenseSum() {
		testMatrixAggregation("ua+", "uar+", "uac+", 0.95);
	}
	
	@Test
	public void testSparseSum() {
		testMatrixAggregation("ua+", "uar+", "uac+", 0.1);
	}
	
	@Test
	public void testDenseSqSum() {
		testMatrixAggregation("uasqk+", "uarsqk+", "uacsqk+", 0.95);
	}
	
	@Test
	public void testSparseSqSum() {
		testMatrixAggregation("uasqk+", "uarsqk+", "uacsqk+", 0.1);
	}
	
	@Test
	public void testDenseMean() {
		testMatrixAggregation("uamean", "uarmean", "uacmean", 0.95);
	}
	
	@Test
	public void testSparseMean() {
		testMatrixAggregation("uamean", "uarmean", "uacmean", 0.1);
	}
	
	@Test
	public void testDenseVar() {
		testMatrixAggregation("uavar", "uarvar", "uacvar", 0.95);
	}
	
	@Test
	public void testSparseVar() {
		testMatrixAggregation("uavar", "uarvar", "uacvar", 0.1);
	}
	
	@Test
	public void testDenseMin() {
		testMatrixAggregation("uamin", "uarmin", "uacmin", 0.95);
	}
	
	@Test
	public void testSparseMin() {
		testMatrixAggregation("uamin", "uarmin", "uacmin", 0.1);
	}
	
	@Test
	public void testDenseMax() {
		testMatrixAggregation("uamax", "uarmax", "uacmax", 0.95);
	}
	
	@Test
	public void testSparseMax() {
		testMatrixAggregation("uamax", "uarmax", "uacmax", 0.1);
	}

	private void testMatrixAggregation(String opcode1, String opcode2, String opcode3, double sp) {
		testMatrixAggregation(getOp(opcode1,1), getOp(opcode1,K), MIN_PAR, 1, sp);
		testMatrixAggregation(getOp(opcode2,1), getOp(opcode2,K), 1, MIN_PAR, sp);
		testMatrixAggregation(getOp(opcode3,1), getOp(opcode3,K), MIN_PAR, 1, sp);
		testMatrixAggregation(getOp(opcode1,1), getOp(opcode1,K), MIN_PAR_SQRT, MIN_PAR_SQRT, sp);
		testMatrixAggregation(getOp(opcode2,1), getOp(opcode2,K), MIN_PAR_SQRT, MIN_PAR_SQRT, sp);
		testMatrixAggregation(getOp(opcode3,1), getOp(opcode3,K), MIN_PAR_SQRT, MIN_PAR_SQRT, sp);
	}

	private void testMatrixAggregation(AggregateUnaryOperator uaop1,
		AggregateUnaryOperator uaopk, int n, int m, double sp)
	{
		MatrixBlock mb1 = MatrixBlock.randOperations(n, m, sp, 0, 0.1, "uniform", 7);
		//run single- and multi-threaded kernels and compare
		MatrixBlock ret1 = mb1.aggregateUnaryOperations(uaop1);
		MatrixBlock ret2 = mb1.aggregateUnaryOperations(uaopk);
		TestUtils.compareMatrices(ret1, ret2, 1e-8);
	}
	
	private AggregateUnaryOperator getOp(String opcode, int threads) {
		return InstructionUtils.parseBasicAggregateUnaryOperator(opcode, threads);
	}
}

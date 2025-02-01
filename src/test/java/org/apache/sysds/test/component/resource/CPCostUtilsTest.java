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

package org.apache.sysds.test.component.resource;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.resource.cost.CPCostUtils;
import org.apache.sysds.resource.cost.VarStats;
import org.apache.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class CPCostUtilsTest {

	@Test
	public void testUnaryNotInstNFLOP() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP("!", -1, -1, expectedValue);
	}

	@Test
	public void testUnaryIsnaInstNFLOP() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.ISNA.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testUnaryIsnanInstNFLOP() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.ISNAN.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testUnaryIsinfInstNFLOP() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.ISINF.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testUnaryCeilInstNFLOP() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.CEIL.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testUnaryFloorInstNFLOP() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.FLOOR.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testAbsInstNFLOPDefaultSparsity() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.ABS.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testAbsInstNFLOPSparse() {
		long expectedValue = (long) (0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.ABS.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testRoundInstNFLOPDefaultSparsity() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.ROUND.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testRoundInstNFLOPSparse() {
		long expectedValue = (long) (0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.ROUND.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testSignInstNFLOPDefaultSparsity() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.SIGN.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testSignInstNFLOPSparse() {
		long expectedValue = (long) (0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.SIGN.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testSpropInstNFLOPDefaultSparsity() {
		long expectedValue = 2 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.SPROP.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testSpropInstNFLOPSparse() {
		long expectedValue = (long) (2 * 0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.SPROP.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testSqrtInstNFLOPDefaultSparsity() {
		long expectedValue = 2 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.SQRT.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testSqrtInstNFLOPSparse() {
		long expectedValue = (long) (2 * 0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.SQRT.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testExpInstNFLOPDefaultSparsity() {
		long expectedValue = 18 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.EXP.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testExpInstNFLOPSparse() {
		long expectedValue = (long) (18 * 0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.EXP.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testSigmoidInstNFLOPDefaultSparsity() {
		long expectedValue = 21 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.SIGMOID.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testSigmoidInstNFLOPSparse() {
		long expectedValue = (long) (21 * 0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.SIGMOID.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testPlogpInstNFLOPDefaultSparsity() {
		long expectedValue = 32 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.PLOGP.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testPlogpInstNFLOPSparse() {
		long expectedValue = (long) (32 * 0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.PLOGP.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testPrintInstNFLOP() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.PRINT.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testAssertInstNFLOP() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.ASSERT.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testSinInstNFLOPDefaultSparsity() {
		long expectedValue = 18 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.SIN.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testSinInstNFLOPSparse() {
		long expectedValue = (long) (18 * 0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.SIN.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testCosInstNFLOPDefaultSparsity() {
		long expectedValue = 22 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.COS.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testCosInstNFLOPSparse() {
		long expectedValue = (long) (22 * 0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.COS.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testTanInstNFLOPDefaultSparsity() {
		long expectedValue = 42 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.TAN.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testTanInstNFLOPSparse() {
		long expectedValue = (long) (42 * 0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.TAN.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testAsinInstNFLOP() {
		long expectedValue = 93 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.ASIN.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testSinhInstNFLOP() {
		long expectedValue = 93 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.SINH.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testAcosInstNFLOP() {
		long expectedValue = 103 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.ACOS.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testCoshInstNFLOP() {
		long expectedValue = 103 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.COSH.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testAtanInstNFLOP() {
		long expectedValue = 40 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.ATAN.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testTanhInstNFLOP() {
		long expectedValue = 40 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.TANH.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testUcumkPlusInstNFLOPDefaultSparsity() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.UCUMKP.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testUcumkPlusInstNFLOPSparse() {
		long expectedValue = (long) (0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.UCUMKP.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testUcumMinInstNFLOPDefaultSparsity() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.UCUMMIN.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testUcumMinInstNFLOPSparse() {
		long expectedValue = (long) (0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.UCUMMIN.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testUcumMaxInstNFLOPDefaultSparsity() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.UCUMMAX.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testUcumMaxInstNFLOPSparse() {
		long expectedValue = (long) (0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.UCUMMAX.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testUcumMultInstNFLOPDefaultSparsity() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.UCUMM.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testUcumMultInstNFLOPSparse() {
		long expectedValue = (long) (0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.UCUMM.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testUcumkPlusMultInstNFLOPDefaultSparsity() {
		long expectedValue = 2 * 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.UCUMKPM.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testUcumkPlusMultInstNFLOPSparse() {
		long expectedValue = (long) (2 * 0.5 * 1000 * 1000);
		testUnaryInstNFLOP(Opcodes.UCUMKPM.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testStopInstNFLOP() {
		long expectedValue = 0;
		testUnaryInstNFLOP(Opcodes.STOP.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testTypeofInstNFLOP() {
		long expectedValue = 1000 * 1000;
		testUnaryInstNFLOP(Opcodes.TYPEOF.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testInverseInstNFLOPDefaultSparsity() {
		long expectedValue = (long) ((4.0 / 3.0) * (1000 * 1000) * (1000 * 1000) * (1000 * 1000));
		testUnaryInstNFLOP(Opcodes.INVERSE.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testInverseInstNFLOPSparse() {
		long expectedValue = (long) ((4.0 / 3.0) * (1000 * 1000) * (0.5 * 1000 * 1000) * (0.5 *1000 * 1000));
		testUnaryInstNFLOP(Opcodes.INVERSE.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testCholeskyInstNFLOPDefaultSparsity() {
		long expectedValue = (long) ((1.0 / 3.0) * (1000 * 1000) * (1000 * 1000) * (1000 * 1000));
		testUnaryInstNFLOP(Opcodes.CHOLESKY.toString(), -1, -1, expectedValue);
	}

	@Test
	public void testCholeskyInstNFLOPSparse() {
		long expectedValue = (long) ((1.0 / 3.0) * (1000 * 1000) * (0.5 * 1000 * 1000) * (0.5 *1000 * 1000));
		testUnaryInstNFLOP(Opcodes.CHOLESKY.toString(), 0.5, 0.5, expectedValue);
	}

	@Test
	public void testLogInstNFLOP() {
		long expectedValue = 32 * 1000 * 1000;
		testBuiltinInstNFLOP(Opcodes.LOG.toString(), -1, expectedValue);
	}

	@Test
	public void testLogNzInstNFLOPDefaultSparsity() {
		long expectedValue = 32 * 1000 * 1000;
		testBuiltinInstNFLOP(Opcodes.LOGNZ.toString(), -1, expectedValue);
	}

	@Test
	public void testLogNzInstNFLOPSparse() {
		long expectedValue = (long) (32 * 0.5 * 1000 * 1000);
		testBuiltinInstNFLOP(Opcodes.LOGNZ.toString(), 0.5, expectedValue);
	}

	@Test
	public void testNrowInstNFLOP() {
		long expectedValue = 10L;
		testAggregateUnaryInstNFLOP(Opcodes.NROW.toString(), expectedValue);
	}

	@Test
	public void testNcolInstNFLOP() {
		long expectedValue = 10L;
		testAggregateUnaryInstNFLOP(Opcodes.NCOL.toString(), expectedValue);
	}

	@Test
	public void testLengthInstNFLOP() {
		long expectedValue = 10L;
		testAggregateUnaryInstNFLOP(Opcodes.LENGTH.toString(), expectedValue);
	}

	@Test
	public void testExistsInstNFLOP() {
		long expectedValue = 10L;
		testAggregateUnaryInstNFLOP(Opcodes.EXISTS.toString(), expectedValue);
	}

	@Test
	public void testLineageInstNFLOP() {
		long expectedValue = 10L;
		testAggregateUnaryInstNFLOP(Opcodes.LINEAGE.toString(), expectedValue);
	}

	@Test
	public void testUakInstNFLOP() {
		long expectedValue = 4 * 1000 * 1000;
		testAggregateUnaryInstNFLOP(Opcodes.UAKP.toString(), expectedValue);
	}

	@Test
	public void testUarkInstNFLOP() {
		long expectedValue = 4L * 2000 * 2000;
		testAggregateUnaryRowInstNFLOP(Opcodes.UARKP.toString(), -1, expectedValue);
		testAggregateUnaryRowInstNFLOP(Opcodes.UARKP.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUackInstNFLOP() {
		long expectedValue = 4L * 3000 * 3000;
		testAggregateUnaryColInstNFLOP(Opcodes.UACKP.toString(), -1, expectedValue);
		testAggregateUnaryColInstNFLOP(Opcodes.UACKP.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUasqkInstNFLOP() {
		long expectedValue = 5L * 1000 * 1000;
		testAggregateUnaryInstNFLOP(Opcodes.UASQKP.toString(), expectedValue);
	}

	@Test
	public void testUarsqkInstNFLOP() {
		long expectedValue = 5L * 2000 * 2000;
		testAggregateUnaryRowInstNFLOP(Opcodes.UARSQKP.toString(), -1, expectedValue);
		testAggregateUnaryRowInstNFLOP(Opcodes.UARSQKP.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUacsqkInstNFLOP() {
		long expectedValue = 5L * 3000 * 3000;
		testAggregateUnaryColInstNFLOP(Opcodes.UACSQKP.toString(), -1, expectedValue);
		testAggregateUnaryColInstNFLOP(Opcodes.UACSQKP.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUameanInstNFLOP() {
		long expectedValue = 7L * 1000 * 1000;
		testAggregateUnaryInstNFLOP(Opcodes.UAMEAN.toString(), expectedValue);
	}

	@Test
	public void testUarmeanInstNFLOP() {
		long expectedValue = 7L * 2000 * 2000;
		testAggregateUnaryRowInstNFLOP(Opcodes.UARMEAN.toString(), -1, expectedValue);
		testAggregateUnaryRowInstNFLOP(Opcodes.UARMEAN.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUacmeanInstNFLOP() {
		long expectedValue = 7L * 3000 * 3000;
		testAggregateUnaryColInstNFLOP(Opcodes.UACMEAN.toString(), -1, expectedValue);
		testAggregateUnaryColInstNFLOP(Opcodes.UACMEAN.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUavarInstNFLOP() {
		long expectedValue = 14L * 1000 * 1000;
		testAggregateUnaryInstNFLOP(Opcodes.UAVAR.toString(), expectedValue);
	}

	@Test
	public void testUarvarInstNFLOP() {
		long expectedValue = 14L * 2000 * 2000;
		testAggregateUnaryRowInstNFLOP(Opcodes.UARVAR.toString(), -1, expectedValue);
		testAggregateUnaryRowInstNFLOP(Opcodes.UARVAR.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUacvarInstNFLOP() {
		long expectedValue = 14L * 3000 * 3000;
		testAggregateUnaryColInstNFLOP(Opcodes.UACVAR.toString(), -1, expectedValue);
		testAggregateUnaryColInstNFLOP(Opcodes.UACVAR.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUamaxInstNFLOP() {
		long expectedValue = 1000 * 1000;
		testAggregateUnaryInstNFLOP(Opcodes.UAMAX.toString(), expectedValue);
	}

	@Test
	public void testUarmaxInstNFLOP() {
		long expectedValue = 2000 * 2000;
		testAggregateUnaryRowInstNFLOP(Opcodes.UARMAX.toString(), -1, expectedValue);
		testAggregateUnaryRowInstNFLOP(Opcodes.UARMAX.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUarimaxInstNFLOP() {
		long expectedValue = 2000 * 2000;
		testAggregateUnaryRowInstNFLOP(Opcodes.UARIMAX.toString(), -1, expectedValue);
		testAggregateUnaryRowInstNFLOP(Opcodes.UARIMAX.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUacmaxInstNFLOP() {
		long expectedValue = 3000 * 3000;
		testAggregateUnaryColInstNFLOP(Opcodes.UACMAX.toString(), -1, expectedValue);
		testAggregateUnaryColInstNFLOP(Opcodes.UACMAX.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUaminInstNFLOP() {
		long expectedValue = 1000 * 1000;
		testAggregateUnaryInstNFLOP(Opcodes.UAMIN.toString(), expectedValue);
	}

	@Test
	public void testUarminInstNFLOP() {
		long expectedValue = 2000 * 2000;
		testAggregateUnaryRowInstNFLOP(Opcodes.UARMIN.toString(), -1, expectedValue);
		testAggregateUnaryRowInstNFLOP(Opcodes.UARMIN.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUariminInstNFLOP() {
		long expectedValue = 2000 * 2000;
		testAggregateUnaryRowInstNFLOP(Opcodes.UARIMIN.toString(), -1, expectedValue);
		testAggregateUnaryRowInstNFLOP(Opcodes.UARIMIN.toString(), 0.5, expectedValue);
	}

	@Test
	public void testUacminInstNFLOP() {
		long expectedValue = 3000 * 3000;
		testAggregateUnaryColInstNFLOP(Opcodes.UACMIN.toString(), -1, expectedValue);
		testAggregateUnaryColInstNFLOP(Opcodes.UACMIN.toString(), 0.5, expectedValue);
	}

	// HELPERS

	private void testUnaryInstNFLOP(String opcode, double sparsityIn, double sparsityOut, long expectedNFLOP) {
		long nnzIn = sparsityIn < 0? -1 : (long) (sparsityIn * 1000 * 1000);
		VarStats input = generateVarStatsMatrix("_mVar1", 1000, 1000, nnzIn);
		long nnzOut = sparsityOut < 0? -1 : (long) (sparsityOut * 1000 * 1000);
		VarStats output = generateVarStatsMatrix("_mVar2", 1000, 1000, nnzOut);

		long result = CPCostUtils.getInstNFLOP(CPType.Unary, opcode, output, input);
		assertEquals(expectedNFLOP, result);
	}

	private void testBuiltinInstNFLOP(String opcode, double sparsityIn, long expectedNFLOP) {
		long nnz = sparsityIn < 0? -1 : (long) (sparsityIn * 1000 * 1000);
		VarStats input = generateVarStatsMatrix("_mVar1", 1000, 1000, nnz);
		VarStats output = generateVarStatsMatrix("_mVar2", 1000, 1000, -1);

		long result = CPCostUtils.getInstNFLOP(CPType.Unary, opcode, output, input);
		assertEquals(expectedNFLOP, result);
	}

	private void testAggregateUnaryInstNFLOP(String opcode, long expectedNFLOP) {
		VarStats input = generateVarStatsMatrix("_mVar1", 1000, 1000, -1);
		VarStats output = generateVarStatsScalarLiteral("_Var2");

		long result = CPCostUtils.getInstNFLOP(CPType.AggregateUnary, opcode, output, input);
		assertEquals(expectedNFLOP, result);
	}

	private void testAggregateUnaryRowInstNFLOP(String opcode, double sparsityOut, long expectedNFLOP) {
		VarStats input = generateVarStatsMatrix("_mVar1", 2000, 1000, -1);
		long nnzOut = sparsityOut < 0? -1 : (long) (sparsityOut * 2000);
		VarStats output = generateVarStatsMatrix("_mVar2", 2000, 1, nnzOut);

		long result = CPCostUtils.getInstNFLOP(CPType.AggregateUnary, opcode, output, input);
		assertEquals(expectedNFLOP, result);
	}

	private void testAggregateUnaryColInstNFLOP(String opcode, double sparsityOut, long expectedNFLOP) {
		VarStats input = generateVarStatsMatrix("_mVar1", 1000, 3000, -1);
		long nnzOut = sparsityOut < 0? -1 : (long) (sparsityOut * 3000);
		VarStats output = generateVarStatsMatrix("_mVar2", 1, 3000, nnzOut);

		long result = CPCostUtils.getInstNFLOP(CPType.AggregateUnary, opcode, output, input);
		assertEquals(expectedNFLOP, result);
	}

	private VarStats generateVarStatsMatrix(String name, long rows, long cols, long nnz) {
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, nnz);
		return new VarStats(name, mc);
	}

	private VarStats generateVarStatsScalarLiteral(String nameOrValue) {
		return new VarStats(nameOrValue, null);
	}
}

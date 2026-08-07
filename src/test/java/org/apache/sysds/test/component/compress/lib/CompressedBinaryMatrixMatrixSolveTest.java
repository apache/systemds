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

package org.apache.sysds.test.component.compress.lib;

import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryMatrixMatrixCPInstruction;
import org.apache.sysds.runtime.matrix.data.LibCommonsMath;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.test.TestUtils;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Drive the solve opcode through {@link BinaryMatrixMatrixCPInstruction} with compressed inputs to cover the
 * commons-math matrix-matrix branch that decompresses compressed left/right operands before solving. The
 * script-level solve tests only ever see uncompressed inputs, so this branch is otherwise unreached.
 */
public class CompressedBinaryMatrixMatrixSolveTest {

	private static final String SOLVE = "solve";

	@BeforeClass
	public static void init() throws java.io.IOException {
		CacheableData.initCaching("compressed_solve_instruction_test");
	}

	@Test
	public void solveCompressedLeftCompressedRight() {
		// A is a compressible, invertible matrix (constant off-diagonal with a larger diagonal), b is a
		// compressed constant right-hand side: both inputs are CompressedMatrixBlock so the instruction must
		// decompress both before delegating to commons-math solve.
		final int n = 200;
		MatrixBlock aUC = new MatrixBlock(n, n, false);
		aUC.allocateDenseBlock();
		for(int i = 0; i < n; i++)
			for(int j = 0; j < n; j++)
				aUC.set(i, j, i == j ? 5.0 : 1.0);
		aUC.recomputeNonZeros();

		CompressedMatrixBlock aC = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(aUC, 1).getLeft();
		assertTrue("A must compress to exercise the compressed-left path", aC instanceof CompressedMatrixBlock);
		CompressedMatrixBlock bC = CompressedMatrixBlockFactory.createConstant(n, 2, 1.0);

		MatrixBlock expected = LibCommonsMath.matrixMatrixOperations(
			CompressedMatrixBlock.getUncompressed(aC), CompressedMatrixBlock.getUncompressed(bC), SOLVE);

		MatrixBlock actual = runSolve(aC, bC);
		TestUtils.compareMatricesBitAvgDistance(expected, actual, 0, 0, SOLVE);
	}

	@Test
	public void solveCompressedLeftDenseRight() {
		// Only the left operand is compressed; the right-hand side stays dense (a single column).
		final int n = 200;
		MatrixBlock aUC = new MatrixBlock(n, n, false);
		aUC.allocateDenseBlock();
		for(int i = 0; i < n; i++)
			for(int j = 0; j < n; j++)
				aUC.set(i, j, i == j ? 5.0 : 1.0);
		aUC.recomputeNonZeros();

		CompressedMatrixBlock aC = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(aUC, 1).getLeft();
		assertTrue("A must compress to exercise the compressed-left path", aC instanceof CompressedMatrixBlock);
		MatrixBlock b = TestUtils.round(TestUtils.generateTestMatrixBlock(n, 1, -5, 5, 1.0, 7));

		MatrixBlock expected = LibCommonsMath.matrixMatrixOperations(
			CompressedMatrixBlock.getUncompressed(aC), b, SOLVE);

		MatrixBlock actual = runSolve(aC, b);
		TestUtils.compareMatricesBitAvgDistance(expected, actual, 0, 0, SOLVE);
	}

	private static MatrixBlock runSolve(MatrixBlock a, MatrixBlock b) {
		ExecutionContext ec = new ExecutionContext(new LocalVariableMap());
		ec.setAutoCreateVars(true);
		ec.setVariable("A", matrixObject("A", a));
		ec.setVariable("b", matrixObject("b", b));
		solveInstruction().processInstruction(ec);
		return ec.getMatrixObject("out").acquireReadAndRelease();
	}

	private static BinaryMatrixMatrixCPInstruction solveInstruction() {
		String in1 = InstructionUtils.concatOperandParts("A", DataType.MATRIX.name(), ValueType.FP64.name(), "false");
		String in2 = InstructionUtils.concatOperandParts("b", DataType.MATRIX.name(), ValueType.FP64.name(), "false");
		String out = InstructionUtils.concatOperandParts("out", DataType.MATRIX.name(), ValueType.FP64.name(), "false");
		String str = InstructionUtils.concatOperands("CP", SOLVE, in1, in2, out);
		return (BinaryMatrixMatrixCPInstruction) BinaryCPInstruction.parseInstruction(str);
	}

	private static MatrixObject matrixObject(String name, MatrixBlock mb) {
		MatrixCharacteristics mc = new MatrixCharacteristics(mb.getNumRows(), mb.getNumColumns(), 1000, mb.getNonZeros());
		MatrixObject mo = new MatrixObject(ValueType.FP64, "/dev/null/" + name,
			new MetaDataFormat(mc, FileFormat.BINARY), mb);
		return mo;
	}
}

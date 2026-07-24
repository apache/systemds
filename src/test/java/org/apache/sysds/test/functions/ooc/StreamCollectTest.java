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

package org.apache.sysds.test.functions.ooc;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.ReorgCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.instructions.ooc.ReblockOOCInstruction;
import org.apache.sysds.runtime.instructions.ooc.ReorgOOCInstruction;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class StreamCollectTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "StreamCollect";
	private final static String TEST_DIR = "functions/ooc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + StreamCollectTest.class.getSimpleName() + "/";
	private final static int rows = 2000;
	private final static int cols = 1000;
	private final static String INPUT_NAME = "input";
	private final static String OUTPUT_NAME = "res";
	private final static double eps = 1e-10;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1);
		addTestConfiguration(TEST_NAME1, config);
	}

	@Test
	public void runRawInstructionSequenceTest() {
		try {
			getAndLoadTestConfiguration(TEST_NAME1);
			MatrixBlock mb = MatrixBlock.randOperations(rows, cols, 1.0, -1, 1, "uniform", 7);
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);
			writer.writeMatrixToHDFS(mb, input(INPUT_NAME), rows, cols, 1000, rows * cols);
			HDFSTool.writeMetaDataFile(input(INPUT_NAME + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, cols, 1000, rows * cols), Types.FileFormat.BINARY);

			ExecutionContext ec = new ExecutionContext(new LocalVariableMap());

			VariableCPInstruction createIn = VariableCPInstruction.parseInstruction(
				"CP°createvar°pREADX°" + input(INPUT_NAME) + "°false°MATRIX°binary°" + rows + "°" + cols +
					"°1000°700147°copy");
			VariableCPInstruction createInRblk = VariableCPInstruction.parseInstruction(
				"CP°createvar°_mVar0°" + input("tmp0") + "°true°MATRIX°binary°" + rows + "°" + cols +
					"°1000°700147°copy");
			ReblockOOCInstruction rblkIn = ReblockOOCInstruction.parseInstruction(
				"OOC°rblk°pREADX·MATRIX·FP64°_mVar0·MATRIX·FP64°1000°true");
			VariableCPInstruction createOut = VariableCPInstruction.parseInstruction(
				"CP°createvar°_mVar1°" + input("tmp1") + "°true°MATRIX°binary°" + rows + "°" + cols +
					"°1000°700147°copy");
			ReorgOOCInstruction oocTranspose = ReorgOOCInstruction.parseInstruction(
				"OOC°r'°_mVar0·MATRIX·FP64°_mVar1·MATRIX·FP64");
			VariableCPInstruction createOut2 = VariableCPInstruction.parseInstruction(
				"CP°createvar°_mVar2°" + input("tmp2") + "°true°MATRIX°binary°" + rows + "°" + cols +
					"°1000°700147°copy");
			ReorgCPInstruction cpTranspose = ReorgCPInstruction.parseInstruction(
				"CP°r'°_mVar1·MATRIX·FP64°_mVar2·MATRIX·FP64°1");

			createIn.processInstruction(ec);
			createInRblk.processInstruction(ec);
			rblkIn.processInstruction(ec);
			createOut.processInstruction(ec);
			oocTranspose.processInstruction(ec);
			createOut2.processInstruction(ec);
			cpTranspose.processInstruction(ec);
		}
		catch(Exception ex) {
			Assert.fail(ex.getMessage());
		}
	}

	@Test
	public void runAppTest1() {
		runAppTest("_1", List.of(List.of(1000D, 1D), List.of(1000D, 1000D)));
	}

	@Test
	public void runAppTest2() {
		runAppTest("_2", List.of(List.of(500D, 1D), List.of(500D, 500D)));
	}

	@Test
	public void runAppTest3() {
		runAppTest("_3", List.of(List.of(1500D, 1D), List.of(1500D, 1500D)));
	}

	private void runAppTest(String scriptSuffix, List<List<Double>> matrixDims) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME1);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + scriptSuffix + ".dml";
			List<String> proArgs = new ArrayList<>();
			proArgs.add("-explain");
			proArgs.add("-stats");
			proArgs.add("-ooc");
			proArgs.add("-args");
			//programArgs = new String[]{"-explain", "-stats", "-ooc",
			//	"-args", input(INPUT_NAME), output(OUTPUT_NAME)};

			for(int i = 0; i < matrixDims.size(); i++) {
				List<Double> dims = matrixDims.get(i);
				int mrows = dims.get(0).intValue();
				int mcols = dims.get(1).intValue();

				// 1. Generate the data as MatrixBlock object
				double[][] A_data = getRandomMatrix(mrows, mcols, 0, 10, 1, 10);

				// 2. Convert the double arrays to MatrixBlock object
				MatrixBlock A_mb = DataConverter.convertToMatrixBlock(A_data);

				// 3. Create a binary matrix writer
				MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);

				// 4. Write matrix A to a binary SequenceFile
				writer.writeMatrixToHDFS(A_mb, input(INPUT_NAME + "_" + i), mrows, mcols, 1000, A_mb.getNonZeros());
				HDFSTool.writeMetaDataFile(input(INPUT_NAME + "_" + i + ".mtd"), Types.ValueType.FP64,
					new MatrixCharacteristics(mrows, mcols, 1000, A_mb.getNonZeros()), Types.FileFormat.BINARY);
				proArgs.add(input(INPUT_NAME + "_" + i));
			}

			proArgs.add(output(OUTPUT_NAME));

			programArgs = proArgs.toArray(String[]::new);

			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1);

			// Validate that at least one OOC instruction was used
			Assert.assertTrue("OOC wasn't used", heavyHittersContainOOC());

			proArgs.remove(2); // Remove ooc flag
			proArgs.set(proArgs.size() - 1, output(OUTPUT_NAME + "_target"));
			programArgs = proArgs.toArray(String[]::new);
			runTest(true, exceptionExpected, null, -1);

			HashMap<MatrixValue.CellIndex, Double> result = readDMLMatrixFromOutputDir(OUTPUT_NAME);
			HashMap<MatrixValue.CellIndex, Double> target = readDMLMatrixFromOutputDir(OUTPUT_NAME + "_target");
			TestUtils.compareMatrices(result, target, eps, "Result", "Target");
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
		}
	}

	private boolean heavyHittersContainOOC() {
		for(String opcode : Statistics.getCPHeavyHitterOpCodes())
			if(opcode.startsWith(Instruction.OOC_INST_PREFIX))
				return true;
		return false;
	}
}

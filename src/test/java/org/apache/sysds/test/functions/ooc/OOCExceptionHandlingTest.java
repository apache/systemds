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
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.io.IOException;

public class OOCExceptionHandlingTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "OOCExceptionHandling";
	private final static String TEST_DIR = "functions/ooc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + OOCExceptionHandlingTest.class.getSimpleName() + "/";
	private static final String INPUT_NAME = "X";
	private static final String INPUT_NAME_2 = "Y";
	private static final String OUTPUT_NAME = "res";

	private final static int rows = 1000;
	private final static int cols = 1000;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1);
		addTestConfiguration(TEST_NAME1, config);
	}

	@Test
	public void runOOCExceptionHandlingTest1() {
		runOOCExceptionHandlingTest(500);
	}

	@Test
	public void runOOCExceptionHandlingTest2() {
		runOOCExceptionHandlingTest(750);
	}


	private void runOOCExceptionHandlingTest(int misalignVals) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME1);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-ooc", "-args", input(INPUT_NAME), input(INPUT_NAME_2), output(OUTPUT_NAME)};

			// 1. Generate the data in-memory as MatrixBlock objects
			double[][] A_data = getRandomMatrix(rows, cols, 1, 2, 1, 7);
			double[][] B_data = getRandomMatrix(rows, 1, 1, 2, 1, 7);

			// 2. Convert the double arrays to MatrixBlock objects
			MatrixBlock A_mb = DataConverter.convertToMatrixBlock(A_data);
			MatrixBlock B_mb = DataConverter.convertToMatrixBlock(B_data);

			// 3. Create a binary matrix writer
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);

			// 4. Write matrix A to a binary SequenceFile

			// Here, we write two faulty matrices which will only be recognized at runtime
			writer.writeMatrixToHDFS(A_mb, input(INPUT_NAME), rows, cols, misalignVals, A_mb.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, cols, 1000, A_mb.getNonZeros()), Types.FileFormat.BINARY);

			writer.writeMatrixToHDFS(B_mb, input(INPUT_NAME_2), rows, 1, 1000, B_mb.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME_2 + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, 1, 1000, B_mb.getNonZeros()), Types.FileFormat.BINARY);

			runTest(true, true, null, -1);
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}

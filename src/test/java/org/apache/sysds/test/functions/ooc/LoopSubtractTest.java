/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.sysds.org/licenses/LICENSE-2.0
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
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.io.IOException;

public class LoopSubtractTest extends AutomatedTestBase {
	private static final String TEST_NAME = "LoopSubtract";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + LoopSubtractTest.class.getSimpleName() + "/";
	private static final double eps = 1e-8;
	private static final String INPUT_NAME_X = "X";
	private static final String INPUT_NAME_Y = "Y";
	private static final String OUTPUT_NAME = "res";

	private static final int rows = 2000;
	private static final int cols = 800;
	private static final double sparsity = 0.8;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME);
		addTestConfiguration(TEST_NAME, config);
	}

	@Test
	public void testLoopSubtractOOC() {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-ooc", "-oocStats",
				"-args", input(INPUT_NAME_X), input(INPUT_NAME_Y), output(OUTPUT_NAME)};

			double[][] X_data = getRandomMatrix(rows, cols, 1, 7, sparsity, 7);
			double[][] Y_data = getRandomMatrix(rows, cols, 0, 1, sparsity, 8);

			MatrixBlock X_mb = DataConverter.convertToMatrixBlock(X_data);
			MatrixBlock Y_mb = DataConverter.convertToMatrixBlock(Y_data);

			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);
			writer.writeMatrixToHDFS(X_mb, input(INPUT_NAME_X), rows, cols, 1000, X_mb.getNonZeros());
			writer.writeMatrixToHDFS(Y_mb, input(INPUT_NAME_Y), rows, cols, 1000, Y_mb.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME_X + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, cols, 1000, X_mb.getNonZeros()), Types.FileFormat.BINARY);
			HDFSTool.writeMetaDataFile(input(INPUT_NAME_Y + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, cols, 1000, Y_mb.getNonZeros()), Types.FileFormat.BINARY);

			OOCCacheManager.getCache().updateLimits(60000000, 100000000);
			runTest(true, false, null, -1);

			programArgs = new String[] {"-explain", "-stats",
				"-args", input(INPUT_NAME_X), input(INPUT_NAME_Y), output(OUTPUT_NAME + "_target")};
			runTest(true, false, null, -1);

			MatrixBlock ret1 = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME),
				Types.FileFormat.BINARY, rows, cols, 1000);
			MatrixBlock ret2 = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME + "_target"),
				Types.FileFormat.BINARY, rows, cols, 1000);
			TestUtils.compareMatrices(ret1, ret2, eps);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}

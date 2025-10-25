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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class CTableTest extends AutomatedTestBase{
	private static final String TEST_NAME1 = "CTableTest";
	private static final String TEST_NAME2 = "WeightedCTableTest";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + CTableTest.class.getSimpleName() + "/";

	private static final String INPUT_NAME1 = "v";
	private static final String INPUT_NAME2 = "w";
	private static final String INPUT_NAME3 = "weights";
	private static final String OUTPUT_NAME = "res";

	private final static double eps = 1e-10;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2));
	}

	@Test
	public void testCTableSimple(){ testCTable(1372, 1012, 5, 5, false);}

	@Test
	public void testCTableValueSetDifferencesNonEmpty(){ testCTable(2000, 37, 4995, 5, false);}

	@Test
	public void testWeightedCTableSimple(){ testCTable(1372, 1012, 5, 5, true);}

	@Test
	public void testWeightedCTableValueSetDifferencesNonEmpty(){ testCTable(2000, 37, 4995, 5, true);}


	public void testCTable(int rows, int cols, int maxValV, int maxValW, boolean isWeighted)
	{
		Types.ExecMode platformOld = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		try {
			String TEST_NAME = isWeighted? TEST_NAME2:TEST_NAME1;

			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			if (isWeighted)
				programArgs = new String[] {"-explain", "-stats", "-ooc", "-args", input(INPUT_NAME1), input(INPUT_NAME2), input(INPUT_NAME3), output(OUTPUT_NAME)};
			else
				programArgs = new String[] {"-explain", "-stats", "-ooc", "-args", input(INPUT_NAME1), input(INPUT_NAME2), output(OUTPUT_NAME)};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			// values <=0 invalid
			double[][] v = TestUtils.floor(getRandomMatrix(rows, cols, 1, maxValV, 1.0, 7));
			double[][] w = TestUtils.floor(getRandomMatrix(rows, cols, 1, maxValW, 1.0, 13));
			double[][] weights = null;

			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);
			writer.writeMatrixToHDFS(DataConverter.convertToMatrixBlock(v), input(INPUT_NAME1), rows, cols, 1000, rows*cols);
			writer.writeMatrixToHDFS(DataConverter.convertToMatrixBlock(w), input(INPUT_NAME2), rows, cols, 1000, rows*cols);

			HDFSTool.writeMetaDataFile(input(INPUT_NAME1+".mtd"), Types.ValueType.FP64, new MatrixCharacteristics(rows,cols,1000,rows*cols), Types.FileFormat.BINARY);
			HDFSTool.writeMetaDataFile(input(INPUT_NAME2+".mtd"), Types.ValueType.FP64, new MatrixCharacteristics(rows,cols,1000,rows*cols), Types.FileFormat.BINARY);

			// for RScript
			writeInputMatrixWithMTD("vR", v, true);
			writeInputMatrixWithMTD("wR", w, true);

			if (isWeighted) {
				weights = TestUtils.floor(getRandomMatrix(rows, cols, 1, maxValW, 1.0, 17));
				writer.writeMatrixToHDFS(DataConverter.convertToMatrixBlock(weights), input(INPUT_NAME3), rows, cols,
					1000, rows * cols);
				HDFSTool.writeMetaDataFile(input(INPUT_NAME3 + ".mtd"), Types.ValueType.FP64,
					new MatrixCharacteristics(rows, cols, 1000, rows * cols), Types.FileFormat.BINARY);
				writeInputMatrixWithMTD("weightsR", weights, true);
			}

			runTest(true, false, null, -1);
			runRScript(true);

			// compare matrices
			HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromExpectedDir("resR");
			double[][] rRes = TestUtils.convertHashMapToDoubleArray(rfile);
			double[][] dmlRes = DataConverter.convertToDoubleMatrix(DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME), Types.FileFormat.BINARY, rRes.length, rRes[0].length, 1000, 1000));

			TestUtils.compareMatrices(rRes, dmlRes, eps);

			String prefix = Instruction.OOC_INST_PREFIX;
			Assert.assertTrue("OOC wasn't used for RBLK",
				heavyHittersContainsString(prefix + Opcodes.RBLK));
		}
		catch(Exception ex) {
			Assert.fail(ex.getMessage());
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}

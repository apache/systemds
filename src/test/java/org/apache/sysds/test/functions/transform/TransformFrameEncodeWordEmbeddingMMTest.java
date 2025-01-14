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


package org.apache.sysds.test.functions.transform;

import static org.apache.sysds.test.functions.transform.TransformFrameEncodeWordEmbedding2Test.generateRandomStrings;
import static org.apache.sysds.test.functions.transform.TransformFrameEncodeWordEmbedding2Test.manuallyDeriveWordEmbeddings;
import static org.apache.sysds.test.functions.transform.TransformFrameEncodeWordEmbedding2Test.shuffleAndMultiplyStrings;
import static org.apache.sysds.test.functions.transform.TransformFrameEncodeWordEmbedding2Test.writeDictToCsvFile;
import static org.apache.sysds.test.functions.transform.TransformFrameEncodeWordEmbedding2Test.writeStringsToCsvFile;

import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class TransformFrameEncodeWordEmbeddingMMTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "TransformFrameEncodeWordEmbeddingsMM";
	private final static String TEST_DIR = "functions/transform/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1));
	}

	@Test
	public void testMultiplication() {
		runMatrixMultiplicationTest(TEST_NAME1, Types.ExecMode.SINGLE_NODE);
	}

	private void runMatrixMultiplicationTest(String testname, Types.ExecMode rt)
	{
		//set runtime platform
		Types.ExecMode rtold = setExecMode(rt);
		try
		{
			int rows = 10;
			int cols = 30;
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();

			// Generate random embeddings for the distinct tokens
			double[][] a = createRandomMatrix("embeddings", rows, cols, 0, 10, 1, new Date().getTime());
			double[][] b = createRandomMatrix("factor", cols, cols, 0, 10, 1, new Date().getTime());

			// Generate random distinct tokens
			List<String> strings = generateRandomStrings(rows, 10);

			// Generate the dictionary by assigning unique ID to each distinct token
			Map<String,Integer> map = writeDictToCsvFile(strings, input(testname + "dict"));

			// Create the dataset by repeating and shuffling the distinct tokens
			int factor = 32;
			rows *= factor;
			List<String> stringsColumn = shuffleAndMultiplyStrings(strings, factor);
			writeStringsToCsvFile(stringsColumn, input(testname + "data"));

			//run script
			programArgs = new String[]{"-stats","-args", input("embeddings"), input(testname + "data"), input(testname + "dict"), input("factor"), output("result")};
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			// Manually derive the expected result
			double[][] res_expected = manuallyDeriveWordEmbeddings(cols, a, map, stringsColumn);
			double[][] res_expectedMM = new double[rows][cols];
			for (int i = 0; i < res_expectedMM.length; i++) {
				for (int j = 0; j < res_expectedMM[i].length; j++) {
					res_expectedMM[i][j] = 0.0;
					for (int k = 0; k < res_expected[i].length; k++) {
						res_expectedMM[i][j] += res_expected[i][k]*b[k][j];
					}
				}
			}
			// Compare results
			HashMap<MatrixValue.CellIndex, Double> res_actual = readDMLMatrixFromOutputDir("result");
			double[][] resultActualDouble = TestUtils.convertHashMapToDoubleArray(res_actual);
			//print2DimDoubleArray(resultActualDouble);
			TestUtils.compareMatrices(res_expectedMM, resultActualDouble, 1e-8);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);

		}
		finally {
			resetExecMode(rtold);
		}
	}
}

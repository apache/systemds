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

import static org.apache.sysds.runtime.functionobjects.KahanPlus.getKahanPlusFnObject;
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
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class TransformFrameEncodeWordEmbeddingRowSumTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "TransformFrameEncodeWordEmbeddingsRowSum";
	private final static String TEST_NAME2 = "TransformFrameEncodeWordEmbeddingsColSum";
	private final static String TEST_NAME3 = "TransformFrameEncodeWordEmbeddingsFullSum";
	private final static String TEST_DIR = "functions/transform/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3));
	}

	@Test
	public void testDedupRowSums() {
		runDedupRowSumTest(TEST_NAME1, Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void testDedupRowSumsSpark() {
		runDedupRowSumTest(TEST_NAME1, Types.ExecMode.SPARK);
	}

	@Test
	public void testDedupColSums() {
		runDedupColSumTest(TEST_NAME2, Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void testDedupFullSums() {
		runDedupFullSumTest(TEST_NAME3, Types.ExecMode.SINGLE_NODE);
	}

	private void runDedupFullSumTest(String testname, Types.ExecMode rt)
	{
		//set runtime platform
		Types.ExecMode rtold = setExecMode(rt);
		try
		{
			int rows = 100;
			int cols = 300;
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();

			// Generate random embeddings for the distinct tokens
			double[][] a = createRandomMatrix("embeddings", rows, cols, 0, 10, 1, new Date().getTime());

			// Generate random distinct tokens
			List<String> strings = generateRandomStrings(rows, 10);

			// Generate the dictionary by assigning unique ID to each distinct token
			Map<String,Integer> map = writeDictToCsvFile(strings, baseDirectory + INPUT_DIR + "dict");

			// Create the dataset by repeating and shuffling the distinct tokens
			List<String> stringsColumn = shuffleAndMultiplyStrings(strings, 320*6);
			writeStringsToCsvFile(stringsColumn, baseDirectory + INPUT_DIR + "data");

			//run script
			programArgs = new String[]{"-stats","-args", input("embeddings"), input("data"), input("dict"), output("result")};
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			// Manually derive the expected result
			double[][] res_expected = manuallyDeriveWordEmbeddings(cols, a, map, stringsColumn);
			double[][] sums_expected = new double[1][1];
			KahanObject ko = new KahanObject(0,0);
			KahanPlus kp = getKahanPlusFnObject();
			for (int i = 0; i < res_expected.length; i++) {
				for (int j = 0; j < res_expected[i].length; j++) {
					kp.execute2(ko,  res_expected[i][j]);
				}
			}
			sums_expected[0][0] = ko._sum;
			// Compare results
			HashMap<MatrixValue.CellIndex, Double> res_actual = readDMLMatrixFromOutputDir("result");
			double[][] resultActualDouble = TestUtils.convertHashMapToDoubleArray(res_actual);
			//print2DimDoubleArray(resultActualDouble);
			TestUtils.compareMatrices(sums_expected, resultActualDouble, 1e-14);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);

		}
		finally {
			resetExecMode(rtold);
		}
	}

	private void runDedupColSumTest(String testname, Types.ExecMode rt)
	{
		//set runtime platform
		Types.ExecMode rtold = setExecMode(rt);
		try
		{
			int rows = 100;
			int cols = 300;
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();

			// Generate random embeddings for the distinct tokens
			double[][] a = createRandomMatrix("embeddings", rows, cols, 0, 10, 1, new Date().getTime());

			// Generate random distinct tokens
			List<String> strings = generateRandomStrings(rows, 10);

			// Generate the dictionary by assigning unique ID to each distinct token
			Map<String,Integer> map = writeDictToCsvFile(strings, baseDirectory + INPUT_DIR + "dict");

			// Create the dataset by repeating and shuffling the distinct tokens
			List<String> stringsColumn = shuffleAndMultiplyStrings(strings, 320*6);
			writeStringsToCsvFile(stringsColumn, baseDirectory + INPUT_DIR + "data");

			//run script
			programArgs = new String[]{"-stats","-args", input("embeddings"), input("data"), input("dict"), output("result")};
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			// Manually derive the expected result
			double[][] res_expected = manuallyDeriveWordEmbeddings(cols, a, map, stringsColumn);
			double[][] sums_expected = new double[1][res_expected[0].length];
			KahanObject ko = new KahanObject(0,0);
			KahanPlus kp = getKahanPlusFnObject();
			for (int i = 0; i < res_expected[0].length; i++) {
				ko.set(0,0);
				for (int j = 0; j < res_expected.length; j++) {
					kp.execute2(ko,  res_expected[j][i]);
				}
				sums_expected[0][i] = ko._sum;
			}
			// Compare results
			HashMap<MatrixValue.CellIndex, Double> res_actual = readDMLMatrixFromOutputDir("result");
			double[][] resultActualDouble = TestUtils.convertHashMapToDoubleArray(res_actual);
			//print2DimDoubleArray(resultActualDouble);
			TestUtils.compareMatrices(sums_expected, resultActualDouble, 1e-9);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);

		}
		finally {
			resetExecMode(rtold);
		}
	}

	private void runDedupRowSumTest(String testname, Types.ExecMode rt)
	{
		//set runtime platform
		Types.ExecMode rtold = setExecMode(rt);
		try
		{
			int rows = 100;
			int cols = 300;
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();

			// Generate random embeddings for the distinct tokens
			double[][] a = createRandomMatrix("embeddings", rows, cols, 0, 10, 1, new Date().getTime());

			// Generate random distinct tokens
			List<String> strings = generateRandomStrings(rows, 10);

			// Generate the dictionary by assigning unique ID to each distinct token
			Map<String,Integer> map = writeDictToCsvFile(strings, baseDirectory + INPUT_DIR + "dict");

			// Create the dataset by repeating and shuffling the distinct tokens
			List<String> stringsColumn = shuffleAndMultiplyStrings(strings, 320);
			writeStringsToCsvFile(stringsColumn, baseDirectory + INPUT_DIR + "data");

			//run script
			programArgs = new String[]{"-stats","-args", input("embeddings"), input("data"), input("dict"), output("result")};
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			// Manually derive the expected result
			double[][] res_expected = manuallyDeriveWordEmbeddings(cols, a, map, stringsColumn);
			double[][] sums_expected = new double[res_expected.length][1];
			KahanObject ko = new KahanObject(0,0);
			KahanPlus kp = getKahanPlusFnObject();
			for (int i = 0; i < res_expected.length; i++) {
				ko.set(0,0);
				for (int j = 0; j < res_expected[i].length; j++) {
					kp.execute2(ko,  res_expected[i][j]);
				}
				sums_expected[i][0] = ko._sum;
			}
			// Compare results
			HashMap<MatrixValue.CellIndex, Double> res_actual = readDMLMatrixFromOutputDir("result");
			double[][] resultActualDouble = TestUtils.convertHashMapToDoubleArray(res_actual);
			//print2DimDoubleArray(resultActualDouble);
			TestUtils.compareMatrices(sums_expected, resultActualDouble, 1e-15);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);

		}
		finally {
			resetExecMode(rtold);
		}
	}
}

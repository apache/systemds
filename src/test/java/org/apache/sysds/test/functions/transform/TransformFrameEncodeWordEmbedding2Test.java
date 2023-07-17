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

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class TransformFrameEncodeWordEmbedding2Test extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "TransformFrameEncodeWordEmbeddings2";
	private final static String TEST_NAME2a = "TransformFrameEncodeWordEmbeddings2MultiCols1";
	private final static String TEST_NAME2b = "TransformFrameEncodeWordEmbeddings2MultiCols2";

	private final static String TEST_DIR = "functions/transform/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1));
		addTestConfiguration(TEST_NAME2a, new TestConfiguration(TEST_DIR, TEST_NAME2a));
		addTestConfiguration(TEST_NAME2b, new TestConfiguration(TEST_DIR, TEST_NAME2b));
	}

	@Test
	public void testTransformToWordEmbeddings() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void testNonRandomTransformToWordEmbeddings2Cols() {
		runTransformTest(TEST_NAME2a, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void testRandomTransformToWordEmbeddings4Cols() {
		runTransformTestMultiCols(TEST_NAME2b, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void runBenchmark(){
		runBenchmark(TEST_NAME1, ExecMode.SINGLE_NODE);
	}


	private void runBenchmark(String testname, ExecMode rt)
	{
		//set runtime platform
		ExecMode rtold = setExecMode(rt);
		try
		{
			int rows = 100;
			//int cols = 300;
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();

			// Generate random embeddings for the distinct tokens
			// double[][] a = createRandomMatrix("embeddings", rows, cols, 0, 10, 1, new Date().getTime());

			// Generate random distinct tokens
			List<String> strings = generateRandomStrings(rows, 10);

			// Generate the dictionary by assigning unique ID to each distinct token
			writeDictToCsvFile(strings, baseDirectory + INPUT_DIR + "dict");

			// Create the dataset by repeating and shuffling the distinct tokens
			List<String> stringsColumn = shuffleAndMultiplyStrings(strings, 320);
			writeStringsToCsvFile(stringsColumn, baseDirectory + INPUT_DIR + "data");

			//run script
			programArgs = new String[]{"-stats","-args", input("embeddings"), input("data"), input("dict"), output("result")};
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);

		}
		finally {
			resetExecMode(rtold);
		}
	}

	private void runTransformTest(String testname, ExecMode rt)
	{
		//set runtime platform
		ExecMode rtold = setExecMode(rt);
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
			List<String> stringsColumn = shuffleAndMultiplyStrings(strings, 32);
			writeStringsToCsvFile(stringsColumn, baseDirectory + INPUT_DIR + "data");

			//run script
			programArgs = new String[]{"-stats","-args", input("embeddings"), input("data"), input("dict"), output("result")};
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			// Manually derive the expected result
			double[][] res_expected = manuallyDeriveWordEmbeddings(cols, a, map, stringsColumn);

			// Compare results
			HashMap<MatrixValue.CellIndex, Double> res_actual = readDMLMatrixFromOutputDir("result");
			double[][] resultActualDouble = TestUtils.convertHashMapToDoubleArray(res_actual);
			TestUtils.compareMatrices(resultActualDouble, res_expected, 1e-6);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);

		}
		finally {
			resetExecMode(rtold);
		}
	}

	@SuppressWarnings("unused")
	private void print2DimDoubleArray(double[][] resultActualDouble) {
		Arrays.stream(resultActualDouble).forEach(
				e -> System.out.println(Arrays.stream(e).mapToObj(d -> String.format("%06.1f", d))
						.reduce("", (sub, elem) -> sub + " " + elem)));
	}

	private void runTransformTestMultiCols(String testname, ExecMode rt)
	{
		//set runtime platform
		ExecMode rtold = setExecMode(rt);
		try
		{
			int rows = 100;
			int cols = 100;
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();

			// Generate random embeddings for the distinct tokens
			double[][] a = createRandomMatrix("embeddings", rows, cols, 0, 10, 1, new Date().getTime());

			// Generate random distinct tokens
			List<String> strings = generateRandomStrings(rows, 10);

			// Generate the dictionary by assigning unique ID to each distinct token
			Map<String,Integer> map = writeDictToCsvFile(strings, baseDirectory + INPUT_DIR + "dict");

			// Create the dataset by repeating and shuffling the distinct tokens
			List<String> stringsColumn = shuffleAndMultiplyStrings(strings, 10);
			writeStringsToCsvFile(stringsColumn, baseDirectory + INPUT_DIR + "data");

			//run script
			programArgs = new String[]{"-stats","-args", input("embeddings"), input("data"), input("dict"), output("result"), output("result2")};
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			// Manually derive the expected result
			double[][] res_expected = manuallyDeriveWordEmbeddings(cols, a, map, stringsColumn);

			// Compare results
			HashMap<MatrixValue.CellIndex, Double> res_actual = readDMLMatrixFromOutputDir("result");
			HashMap<MatrixValue.CellIndex, Double> res_actual2 = readDMLMatrixFromOutputDir("result2");
			double[][] resultActualDouble  = TestUtils.convertHashMapToDoubleArray(res_actual);
			double[][] resultActualDouble2 = TestUtils.convertHashMapToDoubleArray(res_actual2);
			//System.out.println("Actual Result1 [" + resultActualDouble.length + "x" + resultActualDouble[0].length + "]:");
			///print2DimDoubleArray(resultActualDouble);
			//System.out.println("\nActual Result2 [" + resultActualDouble.length + "x" + resultActualDouble[0].length + "]:");
			//print2DimDoubleArray(resultActualDouble2);
			//System.out.println("\nExpected Result [" + res_expected.length + "x" + res_expected[0].length + "]:");
			//print2DimDoubleArray(res_expected);
			TestUtils.compareMatrices(resultActualDouble, res_expected, 1e-6);
			TestUtils.compareMatrices(resultActualDouble, resultActualDouble2, 1e-6);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);

		}
		finally {
			resetExecMode(rtold);
		}
	}

	private double[][] manuallyDeriveWordEmbeddings(int cols, double[][] a, Map<String, Integer> map, List<String> stringsColumn) {
		// Manually derive the expected result
		double[][] res_expected = new double[stringsColumn.size()][cols];
		for (int i = 0; i < stringsColumn.size(); i++) {
			int rowMapped = map.get(stringsColumn.get(i));
			System.arraycopy(a[rowMapped], 0, res_expected[i], 0, cols);
		}
		return res_expected;
	}

	@SuppressWarnings("unused")
	private double[][] generateWordEmbeddings(int rows, int cols) {
		double[][] a = new double[rows][cols];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				a[i][j] = cols *i + j;
			}

		}
		return a;
	}

	public static List<String> shuffleAndMultiplyStrings(List<String> strings, int multiply){
		List<String> out = new ArrayList<>();
		Random random = new Random();
		for (int i = 0; i < strings.size()*multiply; i++) {
			out.add(strings.get(random.nextInt(strings.size())));
		}
		return out;
	}

	public static List<String> generateRandomStrings(int numStrings, int stringLength) {
		List<String> randomStrings = new ArrayList<>();
		Random random = new Random();
		String characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
		for (int i = 0; i < numStrings; i++) {
			randomStrings.add(generateRandomString(random, stringLength, characters));
		}
		return randomStrings;
	}

	public static String generateRandomString(Random random, int stringLength, String characters){
		StringBuilder randomString = new StringBuilder();
		for (int j = 0; j < stringLength; j++) {
			int randomIndex = random.nextInt(characters.length());
			randomString.append(characters.charAt(randomIndex));
		}
		return randomString.toString();
	}

	public static void writeStringsToCsvFile(List<String> strings, String fileName) {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))) {
			for (String line : strings) {
				bw.write(line);
				bw.newLine();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static Map<String,Integer> writeDictToCsvFile(List<String> strings, String fileName) {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))) {
			Map<String,Integer> map = new HashMap<>();
			for (int i = 0; i < strings.size(); i++) {
				map.put(strings.get(i), i);
				bw.write(strings.get(i) + Lop.DATATYPE_PREFIX + (i+1) + "\n");
			}
			return map;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}
}

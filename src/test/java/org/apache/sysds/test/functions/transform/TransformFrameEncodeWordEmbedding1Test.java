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
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.apache.sysds.test.functions.transform.TransformFrameEncodeWordEmbedding2Test.manuallyDeriveWordEmbeddings;
import static org.apache.sysds.test.functions.transform.TransformFrameEncodeWordEmbedding2Test.manuallyDeriveWordEmbeddingsReshape;

public class TransformFrameEncodeWordEmbedding1Test extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "TransformFrameEncodeWordEmbeddings";
	private final static String TEST_NAME2 = "TransformFrameEncodeWordEmbeddings1Reshape";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameEncodeWordEmbedding1Test.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2));
	}

	@Test
	public void testTransformToWordEmbeddings() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testTransformToWordEmbeddingsSpark() {
		runTransformTest(TEST_NAME1, ExecMode.SPARK);
	}

	@Test
	public void testTransformToWordEmbeddingsWithReshape() {
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE);
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

			programArgs = new String[]{"-stats","-args", input("embeddings"), input("data"), input("dict"), output("result")};
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			// Manually derive the expected result
			double[][] res_expected = testname.equals(TEST_NAME2) ? manuallyDeriveWordEmbeddingsReshape(cols, a, map, stringsColumn, 10) : manuallyDeriveWordEmbeddings(cols, a, map, stringsColumn);

			// Compare results
			HashMap<MatrixValue.CellIndex, Double> res_actual = readDMLMatrixFromOutputDir("result");
			TestUtils.compareMatrices(TestUtils.convertHashMapToDoubleArray(res_actual), res_expected, 1e-6);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(rtold);
		}
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

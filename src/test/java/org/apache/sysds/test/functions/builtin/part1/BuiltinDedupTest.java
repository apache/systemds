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

package org.apache.sysds.test.functions.builtin.part1;

import java.util.Arrays;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

//import org.junit.runner.RunWith;
//import org.junit.runners.Parameterized;

//@RunWith(value = Parameterized.class)
//@net.jcip.annotations.NotThreadSafe
public class BuiltinDedupTest extends AutomatedTestBase {
	private final static String TEST_NAME = "distributed_representation"; 
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDedupTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"Y_unique", "Y_duplicates"}));
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@Test
	public void testSimpleDedupCP() {
		runTestCase(ExecType.CP);
	}

	private void runTestCase(ExecType execType) {
		Types.ExecMode platformOld = setExecMode(execType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			programArgs = new String[]{
				"-stats", "-args",
				input("X"), input("gloveMatrix"), input("vocab"),
				"cosine", "0.8", 
				output("Y_unique"), output("Y_duplicates")
			};

			// ----- Frame X -----
			String[][] X = new String[][]{
				{"John Doe", "New York"},
				{"Jon Doe", "New York City"},
				{"Jane Doe", "Boston"},
				{"John Doe", "NY"}
			};
			ValueType[] schemaX = new ValueType[]{ValueType.STRING, ValueType.STRING};
			FrameBlock fbX = new FrameBlock(schemaX);
			for (String[] row : X) fbX.appendRow(row);
			writeInputFrameWithMTD("X", fbX, true, new MatrixCharacteristics(X.length, X[0].length, -1, -1), schemaX, FileFormat.BINARY);

			// ----- Vocab -----
			String[][] vocab = new String[][]{
				{"john"}, {"doe"}, {"new"}, {"york"},
				{"city"}, {"boston"}, {"ny"}, {"jane"}
			};
			ValueType[] schemaVocab = new ValueType[]{ValueType.STRING};
			FrameBlock fbVocab = new FrameBlock(schemaVocab);
			for (String[] row : vocab) fbVocab.appendRow(row);
			writeInputFrameWithMTD("vocab", fbVocab, true, new MatrixCharacteristics(vocab.length, 1, -1, -1), schemaVocab, FileFormat.BINARY);

			// ----- Glove-Matrix -----
			double[][] gloveMatrix = getRandomMatrix(vocab.length, 50, -0.5, 0.5, 1, 123);
			writeInputMatrixWithMTD("gloveMatrix", gloveMatrix, true);

			// Run
			runTest(true, false, null, -1);

			// Expected unique
			String[][] expectedUnique = new String[][] {
				{"John Doe", "New York"},
				{"Jon Doe", "New York City"},
				{"Jane Doe", "Boston"}
			};

			// Expected duplicates
			String[][] expectedDupes = new String[][] {
				{"John Doe", "NY"}
			};

			/*
			// --- Validate output frames ---
			FrameBlock outUnique = readDMLFrameFromHDFS("Y_unique", FileFormat.BINARY);
			FrameBlock outDupes = readDMLFrameFromHDFS("Y_duplicates", FileFormat.BINARY);

			String[][] actualUnique = frameBlockToStringArray(outUnique);
			String[][] actualDupes = frameBlockToStringArray(outDupes);

			// Compare
			System.out.println("Unqiue tuples: " + Arrays.deepToString(actualUnique));
			System.out.println("Actual Dupes: " + Arrays.deepToString(actualDupes));
			assertStringArrayEquals(expectedUnique, actualUnique);
			assertStringArrayEquals(expectedDupes, actualDupes);
			*/

		}
		catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
		finally {
			rtplatform = platformOld;
		}
	}

	private String[][] frameBlockToStringArray(FrameBlock fb) {
		int rows = fb.getNumRows();
		int cols = fb.getNumColumns();
		String[][] out = new String[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				Object val = fb.get(i, j);
				out[i][j] = (val != null) ? val.toString() : "";
			}
		}
		return out;
	}	

	private void assertStringArrayEquals(String[][] expected, String[][] actual) {
		if (expected.length != actual.length || expected[0].length != actual[0].length) {
			throw new AssertionError("Array dimensions do not match");
		}
		for (int i = 0; i < expected.length; i++) {
			for (int j = 0; j < expected[0].length; j++) {
				if (!expected[i][j].equals(actual[i][j])) {
					throw new AssertionError(String.format(
						"Mismatch at [%d,%d]: expected='%s' but got='%s'",
						i, j, expected[i][j], actual[i][j]
					));
				}
			}
		}
	}
}

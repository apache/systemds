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
import java.util.Collection;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class BuiltinDedupTest extends AutomatedTestBase {
	private final static String TEST_NAME = "distributed_representation";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDedupTest.class.getSimpleName() + "/";

	@Parameterized.Parameter()
	public String similarityMeasure;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{"cosine"},
			{"euclidean"}
		});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"Y"}));
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@Test
	public void testDedupCP() {
		runDedupTests(ExecType.CP);
	}

	@Test
	public void testDedupSPARK() {
		runDedupTests(ExecType.SPARK);
	}

	private void runDedupTests(ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args", input("X"), input("gloveMatrix"), input("vocab"), similarityMeasure, 
				"0.8", output("Y")};

			// Mock input data
			String[][] X = new String[][]{
				{"John Doe", "New York"},
				{"Jon Doe", "New York City"},
				{"Jane Doe", "Boston"},
				{"John Doe", "NY"}
			};

			// Mock gloveMatrix embeddings
			double[][] gloveMatrix = getRandomMatrix(5, 50, -0.5, 0.5, 1, 123);

			// Mock vocabulary
			String[][] vocab = new String[][]{
				{"john"},
				{"doe"},
				{"new"},
				{"york"},
				{"city"}
			};

			writeInputFrameWithMTD("X", X, false);
			writeInputMatrixWithMTD("gloveMatrix", gloveMatrix, true);
			writeInputFrameWithMTD("vocab", vocab, false);

			runTest(true, false, null, -1);

			// You can add assertions to check results if expected results are defined.
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			rtplatform = platformOld;
		}
	}
}

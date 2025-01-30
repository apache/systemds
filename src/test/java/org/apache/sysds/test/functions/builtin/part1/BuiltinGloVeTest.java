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

import java.io.IOException;
import java.util.Objects;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

public class BuiltinGloVeTest extends AutomatedTestBase {

	private static final String TEST_NAME = "glove";
	private static final String TEST_DIR = "functions/builtin/";
	private static final String RESOURCE_DIRECTORY = "./src/test/resources/datasets/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinGloVeTest.class.getSimpleName() + "/";

	private static final int TOP_K = 5;
	private static final double ACCURACY_THRESHOLD = 0.85;
	
	private static final double seed = 45;
	private static final double vector_size = 100;
	private static final double alpha = 0.75;
	private static final double eta = 0.05;
	private static final double x_max = 100;
	private static final double tol = 1e-4;
	private static final double iterations = 10000;
	private static final double print_loss_it  = 100;
	private static final double maxTokens = 2600;
	private static final double windowSize = 15;
	private static final String distanceWeighting = "TRUE";
	private static final String symmetric = "TRUE";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"out_result"}));
	}

	@Test
	public void gloveTest() throws IOException{
		// Using top-5 words for similarity comparison
		runGloVe(TOP_K); 

		// Read the computed similarity results from SystemDS
		FrameBlock computedSimilarity = readDMLFrameFromHDFS("out_result", FileFormat.CSV);

		// Load expected results (precomputed in Python)
		FrameBlock expectedSimilarity = readDMLFrameFromHDFS(RESOURCE_DIRECTORY + "/GloVe/gloveExpectedTop10.csv", FileFormat.CSV, false);

		// Compute accuracy by comparing computed and expected results
		double accuracy = computeAccuracy(computedSimilarity, expectedSimilarity, TOP_K);

		System.out.println("Computed Accuracy: " + accuracy);

		// Ensure accuracy is above a reasonable threshold
		assert accuracy > ACCURACY_THRESHOLD : "Accuracy too low! Expected > 85% match.";
	}

	public void runGloVe(int topK) {
		// Load test configuration
		Types.ExecMode platformOld = setExecMode(Types.ExecType.CP);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			programArgs = new String[] {
					"-nvargs",
					"input=" + RESOURCE_DIRECTORY + "20news/20news_subset_untokenized.csv",
					"seed=" + seed, 
					"vector_size=" + vector_size, 
					"alpha=" + alpha, 
					"eta=" + eta, 
					"x_max=" + x_max, 
					"tol=" + tol, 
					"iterations=" + iterations, 
					"print_loss_it=" + print_loss_it, 
					"maxTokens=" + maxTokens, 
					"windowSize=" + windowSize, 
					"distanceWeighting=" + distanceWeighting,
					"symmetric=" + symmetric,
					"topK=" + topK,
					"out_result=" + output("out_result")
			};

			System.out.println("Running DML script...");
			runTest(true, false, null, -1);
			System.out.println("Test execution completed.");
		} finally {
			rtplatform = platformOld;
		}
	}

	/**
	 * Computes accuracy by comparing top-K similar words between computed and expected results.
	 */
	private double computeAccuracy(FrameBlock computed, FrameBlock expected, int k) {
		int count = 0;
		for (int i = 0; i < computed.getNumRows(); i++) {
			int matchCount = 0;
			for (int j = 1; j < k; j++) {
				String word1 = computed.getString(i, j);
				for (int m = 0; m < k; m++) {
					if (Objects.equals(word1, expected.getString(i, m))) {
						matchCount++;
						break;
					}
				}
			}
			if (matchCount > 0) {
				count++;
			}
		}
		return (double) count / computed.getNumRows();
	}
}

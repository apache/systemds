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

import java.io.IOException;
import java.util.Random;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class KMeansTest extends AutomatedTestBase {
	private static final String TEST_NAME = "KMeans";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + KMeansTest.class.getSimpleName() + "/";

	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME_OOC = "C";
	private static final String OUTPUT_NAME_CP = "C_target";

	private static final int ROWS = 10000;
	private static final int COLS = 400;
	private static final int K = 8;
	private static final int RUNS = 3;
	private static final int MAX_ITER = 50;
	private static final int SEED = 7;
	private static final int BLOCK_SIZE = 1000;
	private static final double MAX_VAL = 2;
	private static final double SPARSITY_DENSE = 1.0;
	private static final double SPARSITY_SPARSE = 0.2;
	private static final double EPS = 1e-9;
	private static final double CLUSTER_NOISE = 0.05;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	@Test
	public void testKMeansDense() {
		runKMeansTest(false);
	}

	@Test
	public void testKMeansSparse() {
		runKMeansTest(true);
	}

	private void runKMeansTest(boolean sparse) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String home = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = home + TEST_NAME + ".dml";

			double[][] xData = generateClusteredInput(sparse);
			writeBinaryWithMTD(INPUT_NAME, DataConverter.convertToMatrixBlock(xData));

			programArgs = new String[] {"-explain", "-stats", "-ooc", "-args",
				input(INPUT_NAME), Integer.toString(K), Integer.toString(RUNS), Integer.toString(MAX_ITER),
				Double.toString(EPS), Integer.toString(SEED), output(OUTPUT_NAME_OOC)};
			runTest(true, false, null, -1);

			programArgs = new String[] {"-explain", "-stats", "-args",
				input(INPUT_NAME), Integer.toString(K), Integer.toString(RUNS), Integer.toString(MAX_ITER),
				Double.toString(EPS), Integer.toString(SEED), output(OUTPUT_NAME_CP)};
			runTest(true, false, null, -1);

			MatrixBlock centersOOC = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME_OOC),
				Types.FileFormat.BINARY, K, COLS, BLOCK_SIZE);
			MatrixBlock centersCP = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME_CP),
				Types.FileFormat.BINARY, K, COLS, BLOCK_SIZE);

			TestUtils.compareMatrices(centersOOC, centersCP, EPS);
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
		}
	}

	private static double[][] generateClusteredInput(boolean sparse) {
		Random rand = new Random(SEED);
		double[][] centers = new double[K][COLS];
		for(int k = 0; k < K; k++) {
			for(int c = 0; c < COLS; c++)
				centers[k][c] = rand.nextDouble() * MAX_VAL;
		}

		double[][] data = new double[ROWS][COLS];
		double keepProb = sparse ? SPARSITY_SPARSE : SPARSITY_DENSE;
		for(int r = 0; r < ROWS; r++) {
			int cluster = r % K;
			for(int c = 0; c < COLS; c++) {
				double v = centers[cluster][c] + rand.nextGaussian() * CLUSTER_NOISE;
				v = Math.max(0, Math.min(MAX_VAL, v));
				if(rand.nextDouble() > keepProb)
					v = 0;
				data[r][c] = v;
			}
		}
		return data;
	}
}

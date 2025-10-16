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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;
import java.util.Random;

public class lmDSTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "lmDS";
	private final static String TEST_DIR = "functions/ooc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + lmDSTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	private static final String INPUT_NAME = "X";
	private static final String INPUT_NAME2 = "y";
	private static final String OUTPUT_NAME = "R";

	private final static int rows = 100000;
	private final static int cols_wide = 500;
	private final static int cols_skinny = 500;

	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1);
		addTestConfiguration(TEST_NAME1, config);
	}

	@Test
	public void testlmDS1() {
		runMatrixVectorMultiplicationTest(cols_wide, false);
	}

	@Test
	public void testlmDS2() {
		runMatrixVectorMultiplicationTest(cols_skinny, false);
	}

	private void runMatrixVectorMultiplicationTest(int cols, boolean sparse )
	{
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try
		{
			getAndLoadTestConfiguration(TEST_NAME1);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-explain", "-stats", "-ooc",
					"-args", input(INPUT_NAME), input(INPUT_NAME2), output(OUTPUT_NAME)};

			// 1. Generate the data in-memory as MatrixBlock objects
			double[][] A_data = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 7);
//			double[][] A_data = generateFullRankMatrix(rows, cols, 10L);
			double[][] x_data = getRandomMatrix(rows, 1, 0, 1, 1.0, 3);
//			double[][] x_data = getRandomMatrix(rows, 1, 0, 1, 1.0, 20L);

			// 2. Convert the double arrays to MatrixBlock objects
			MatrixBlock A_mb = DataConverter.convertToMatrixBlock(A_data);
			MatrixBlock x_mb = DataConverter.convertToMatrixBlock(x_data);

			// 3. Create a binary matrix writer
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);

			// 4. Write matrix A to a binary SequenceFile
			writer.writeMatrixToHDFS(A_mb, input(INPUT_NAME), rows, cols, 1000, A_mb.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME + ".mtd"), Types.ValueType.FP64,
					new MatrixCharacteristics(rows, cols, 1000, A_mb.getNonZeros()), Types.FileFormat.BINARY);

			// 5. Write vector x to a binary SequenceFile
			writer.writeMatrixToHDFS(x_mb, input(INPUT_NAME2), rows, 1, 1000, x_mb.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME2 + ".mtd"), Types.ValueType.FP64,
					new MatrixCharacteristics(rows, 1, 1000, x_mb.getNonZeros()), Types.FileFormat.BINARY);

			fullRScriptName = HOME + TEST_NAME1 + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " "  + expectedDir();

			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1);
//			runRScript(true);

//			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir(OUTPUT_NAME);

			double[][] C1 = readMatrix(output(OUTPUT_NAME), Types.FileFormat.BINARY, rows, cols, 1000, 1000);
			double result = 0.0;
			for(int i = 0; i < 100; i++) { // verify the results with Java
				double expected = 0.0;
				for(int j = 0; j < 100; j++) {
					expected += A_mb.get(i, j) * x_mb.get(j,0);
				}
				result = C1[i][0];
				System.out.println("(i): " + i + " ->> expected" + expected + ", result: " + result);
//				Assert.assertEquals(expected, result, eps);
			}
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
		}
	}

	private static double[][] readMatrix(String fname, Types.FileFormat fmt, long rows, long cols, int brows, int bcols )
			throws IOException
	{
		MatrixBlock mb = DataConverter.readMatrixFromHDFS(fname, fmt, rows, cols, brows, bcols);
		double[][] C = DataConverter.convertToDoubleMatrix(mb);
		return C;
	}

	/**
	 * Generates a matrix that is guaranteed to have full column rank,
	 * preventing a singular t(X)%*%X matrix.
	 *
	 * @param rows Number of rows
	 * @param cols Number of columns (must be <= rows)
	 * @param seed Random seed
	 * @return A new double[][] matrix
	 */
	private double[][] generateFullRankMatrix(int rows, int cols, long seed) {
		if (cols > rows) {
			throw new IllegalArgumentException("For a full-rank matrix, cols must be <= rows.");
		}
		double[][] A = new double[rows][cols];
		Random rand = new Random(seed);

		// 1. Create a dominant diagonal by starting with an identity-like structure
		for (int i = 0; i < cols; i++) {
			A[i][i] = 1.0;
		}

		// 2. Add small random noise to all other elements to ensure non-singularity
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				if (i != j) { // Don't overwrite the dominant diagonal
					A[i][j] = rand.nextDouble() * 0.1; // Small noise
				}
			}
		}
		return A;
	}
}

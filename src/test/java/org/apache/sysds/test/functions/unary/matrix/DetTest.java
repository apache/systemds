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

package org.apache.sysds.test.functions.unary.matrix;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.HashMap;


public class DetTest extends AutomatedTestBase {

	private static final String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + DetTest.class.getSimpleName() + "/";
	private static final String DML_SCRIPT_NAME = "DetTest";
	private static final String R_SCRIPT_NAME = "DetTest";

	private static final String TEST_NAME_WRONG_DIM = "WrongDimensionsTest";
	private static final String TEST_NAME_DET_TEST = "DetTest";

	// The number of rows and columns should not be chosen to be too large,
	// because the calculation of the determinant can introduce rather large
	// floating point errors with large row sizes, because there are many
	// floating point operations involving both multiplication and addition.
	private final static int rows = 23;
	private final static double _sparsityDense = 0.7;
	private final static double _sparsitySparse = 0.2;
	private final static double eps = 1e-8;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME_WRONG_DIM, new TestConfiguration(TEST_CLASS_DIR, DML_SCRIPT_NAME, new String[] { "d" }));
		addTestConfiguration(TEST_NAME_DET_TEST, new TestConfiguration(TEST_CLASS_DIR, DML_SCRIPT_NAME, new String[] { "d" }) );
	}

	@Test
	public void testWrongDimensions() {
		int wrong_rows = 10;
		int wrong_cols = 9;
		
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME_WRONG_DIM);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + DML_SCRIPT_NAME + ".dml";
		programArgs = new String[]{"-args", input("A"), output("d") };

		double[][] A = getRandomMatrix(wrong_rows, wrong_cols, -1, 1, 0.5, 3);
		writeInputMatrixWithMTD("A", A, true);
		runTest(true, true, LanguageException.class, -1);
	}

	@Test
	public void testDetMatrixDense() {
		runDetTest(false);
	}

	@Test
	public void testDetMatrixSparse() {
		runDetTest(true);
	}

	private void runDetTest(boolean sparse) {
		ExecMode platformOld = rtplatform;
		rtplatform = ExecMode.HYBRID;
		
		try {
			double sparsity = (sparse) ? _sparsitySparse : _sparsityDense;
			getAndLoadTestConfiguration(TEST_NAME_DET_TEST);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + DML_SCRIPT_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), output("d")};
			
			fullRScriptName = HOME + R_SCRIPT_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			double[][] A = getRandomMatrix(rows, rows, -1, 1, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);
	
			runTest(true, false, null, -1);
			runRScript(true);
		
			HashMap<CellIndex, Double> dmlfile = readDMLScalarFromOutputDir("d");
			HashMap<CellIndex, Double> rfile  = readRScalarFromExpectedDir("d");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}


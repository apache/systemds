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

package org.apache.sysds.test.functions.builtin;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.HashMap;

public class BuiltinImageBrightnessTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "image_brightness";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinImageBrightnessTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;
	private final static int rows = 256;
	private final static int cols = 256;
	private final static double spSparse = 0.1;
	private final static double spDense = 0.9;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testImageBrightnessMatrixDenseCP() {runImageBrightnessTest(false, ExecType.CP);
	}

	@Test
	public void testImageBrightnessMatrixSparseCP() {runImageBrightnessTest(true, ExecType.CP);
	}

	@Test
	public void testImageBrightnessMatrixDenseSP() {runImageBrightnessTest(false, ExecType.SPARK);
	}

	@Test
	public void testImageBrightnessMatrixSparseSP() {runImageBrightnessTest(false,ExecType.SPARK);
	}

	private void runImageBrightnessTest(boolean sparse, ExecType instType)
	{
		ExecMode platformOld = setExecMode(instType);
		
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			double sparsity = sparse ? spSparse : spDense;

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-nvargs",
					"in_file=" + input("A"),
					"out_file=" + output("B"),
			};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 255, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

		}
		finally {
			rtplatform = platformOld;
		}
	}
}

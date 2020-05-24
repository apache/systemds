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

public class BuiltinImageMirrorTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "image_mirror";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinImageMirrorTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;
	private final static int rows = 256;
	private final static int cols = 256;
	private final static double spSparse = 0.1;
	private final static double spDense = 0.9;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"Bx", "By"}));
	}

	@Test
	public void testImageMirrorMatrixDenseCP() {runImageMirrorTest(false, ExecType.CP);
	}

	@Test
	public void testImageMirrorMatrixSparseCP() {runImageMirrorTest(true, ExecType.CP);
	}

	@Test
	public void testImageMirrorMatrixDenseSP() {runImageMirrorTest(false, ExecType.SPARK);
	}

	@Test
	public void testImageMirrorMatrixSparseSP() {runImageMirrorTest(false,ExecType.SPARK);
	}

	private void runImageMirrorTest(boolean sparse, ExecType instType)
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
					"x_out_file=" + output("Bx"),
					"y_out_file=" + output("By"),
			};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 255, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices of the image mirrored on the x axis
			HashMap<MatrixValue.CellIndex, Double> dmlfile_x = readDMLMatrixFromHDFS("Bx");
			HashMap<MatrixValue.CellIndex, Double> rfile_x  = readRMatrixFromFS("Bx");
			TestUtils.compareMatrices(dmlfile_x, rfile_x, eps, "Stat-DML", "Stat-R");

			//compare matrices of the image mirrored on the y axis
			HashMap<MatrixValue.CellIndex, Double> dmlfile_y = readDMLMatrixFromHDFS("By");
			HashMap<MatrixValue.CellIndex, Double> rfile_y  = readRMatrixFromFS("By");
			TestUtils.compareMatrices(dmlfile_y, rfile_y, eps, "Stat-DML", "Stat-R");
		}

		finally {
			rtplatform = platformOld;
		}
	}
}

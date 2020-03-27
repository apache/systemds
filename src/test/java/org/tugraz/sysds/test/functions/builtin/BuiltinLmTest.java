/*
 * Copyright 2018 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.functions.builtin;

import org.junit.Test;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

import java.util.HashMap;

public class BuiltinLmTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "lm";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinLmTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;
	private final static int rows = 10;
	private final static int cols = 3;
	private final static double spSparse = 0.3;
	private final static double spDense = 0.7;

	public enum LinregType {
		CG, DS, AUTO
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}

	@Test
	public void testLmMatrixDenseCPlm() {
		runLmTest(false, ExecType.CP, LinregType.AUTO);
	}

	@Test
	public void testLmMatrixSparseCPlm() {
		runLmTest(true, ExecType.CP, LinregType.AUTO);
	}

	@Test
	public void testLmMatrixDenseSPlm() {
		runLmTest(false, ExecType.SPARK, LinregType.AUTO);
	}

	@Test
	public void testLmMatrixSparseSPlm() {
		runLmTest(true, ExecType.SPARK, LinregType.AUTO);
	}

	@Test
	public void testLmMatrixDenseCPlmDS() {
		runLmTest(false, ExecType.CP, LinregType.DS);
	}

	@Test
	public void testLmMatrixSparseCPlmDS() {
		runLmTest(true, ExecType.CP, LinregType.DS);
	}

	@Test
	public void testLmMatrixDenseSPlmDS() {
		runLmTest(false, ExecType.SPARK, LinregType.DS);
	}

	@Test
	public void testLmMatrixSparseSPlmDS() {
		runLmTest(true, ExecType.SPARK, LinregType.DS);
	}

	@Test
	public void testLmMatrixDenseCPlmCG() {
		runLmTest(false, ExecType.CP, LinregType.CG);
	}

	@Test
	public void testLmMatrixSparseCPlmCG() {
		runLmTest(true, ExecType.CP, LinregType.CG);
	}

	@Test
	public void testLmMatrixDenseSPlmCG() {
		runLmTest(false, ExecType.SPARK, LinregType.CG);
	}

	@Test
	public void testLmMatrixSparseSPlmCG() {
		runLmTest(true, ExecType.SPARK, LinregType.CG);
	}

	private void runLmTest(boolean sparse, ExecType instType, LinregType linregAlgo)
	{
		ExecMode platformOld = setExecMode(instType);
		
		String dml_test_name = TEST_NAME;
		switch (linregAlgo) {
			case AUTO: break;
			case DS: dml_test_name += "DS"; break;
			case CG: dml_test_name += "CG"; break;
		}

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			double sparsity = sparse ? spSparse : spDense;

			String HOME = SCRIPT_DIR + TEST_DIR;


			fullDMLScriptName = HOME + dml_test_name + ".dml";
			programArgs = new String[]{"-explain", "-stats", "-args", input("A"), input("B"), output("C") };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " "  + expectedDir();

			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(rows, 1, 0, 10, 1.0, 3);
			writeInputMatrixWithMTD("B", B, true);

			runTest(true, false, null, -1);
			runRScript(true); 

			//compare matrices 

			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}

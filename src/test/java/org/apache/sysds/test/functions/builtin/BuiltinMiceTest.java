/*
 * Copyright 2020 Graz University of Technology
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

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.lops.LopProperties;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

import java.util.HashMap;

public class BuiltinMiceTest extends AutomatedTestBase {
	private final static String TEST_NAME = "mice";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinMiceTest.class.getSimpleName() + "/";

	private final static String DATASET = SCRIPT_DIR +"functions/transform/input/ChickWeight.csv";
	private final static double eps = 0.2;
	private final static int iter = 3;
	private final static int com = 2;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
	}
	@Test
	public void testMiceCP() {
		runMiceNominalTest( LopProperties.ExecType.CP);
	}

//	@Test
//	public void testMiceSpark() {
//		runMiceNominalTest( LopProperties.ExecType.SPARK);
//	}


	private void runMiceNominalTest( LopProperties.ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			double[][] mask = {{ 0.0, 0.0, 1.0, 1.0, 0.0}};
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-nvargs", "X=" + DATASET, "Mask="+input("M"), "iteration=" + iter,  "com=" + com, "dataN=" + output("N"), "dataC=" + output("C")};
			writeInputMatrixWithMTD("M", mask, true);

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " +DATASET+ " " +inputDir() + " "  + expectedDir();

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfileN = readDMLMatrixFromHDFS("N");
			HashMap<MatrixValue.CellIndex, Double> rfileN  = readRMatrixFromFS("N");

			HashMap<MatrixValue.CellIndex, Double> dmlfileC = readDMLMatrixFromHDFS("C");
			HashMap<MatrixValue.CellIndex, Double> rfileC  = readRMatrixFromFS("C");
			// compare numerical imputations
			TestUtils.compareMatrices(dmlfileN, rfileN, eps, "Stat-DML", "Stat-R");
			// compare categorical imputations
			int countTrue = 0;
			for (MatrixValue.CellIndex index : dmlfileC.keySet()) {
				Double v1 = dmlfileC.get(index);
				Double v2 = rfileC.get(index);
				if(v1.equals(v2))
					countTrue++;
				}

			if(countTrue / (double)dmlfileC.size() > 0.98)
				Assert.assertTrue(true);
			else
				Assert.fail();
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
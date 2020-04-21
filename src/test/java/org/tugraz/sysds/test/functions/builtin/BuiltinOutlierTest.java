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

import java.util.HashMap;

import org.junit.Test;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class BuiltinOutlierTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "outlier";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinOutlierTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-8;
	private final static int rows = 1765;
	private final static int cols = 392;
	private final static double spDense = 0.7;
	private final static double spSparse = 0.1;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}

	@Test
	public void testOutlierDensePosCP() {
		runOutlierTest(false, false, ExecType.CP);
	}
	
	@Test
	public void testOutlierDensePosSP() {
		runOutlierTest(false, false, ExecType.SPARK);
	}
	
	@Test
	public void testOutlierDenseNegCP() {
		runOutlierTest(false, true, ExecType.CP);
	}
	
	@Test
	public void testOutlierDenseNegSP() {
		runOutlierTest(false, true, ExecType.SPARK);
	}
	
	@Test
	public void testOutlierSparsePosCP() {
		runOutlierTest(true, false, ExecType.CP);
	}
	
	@Test
	public void testOutlierSparsePosSP() {
		runOutlierTest(true, false, ExecType.SPARK);
	}
	
	@Test
	public void testOutlierSparseNegCP() {
		runOutlierTest(true, true, ExecType.CP);
	}
	
	@Test
	public void testOutlierSparseNegSP() {
		runOutlierTest(true, true, ExecType.SPARK);
	}
	

	private void runOutlierTest(boolean sparse, boolean opposite, ExecType instType)
	{
		ExecMode platformOld = setExecMode(instType);
		
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", input("A"),
				String.valueOf(opposite).toUpperCase(), output("B") };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " 
				+ String.valueOf(opposite).toUpperCase() + " " + expectedDir();
			
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -1, 1, sparse?spSparse:spDense, 7);
			writeInputMatrixWithMTD("A", A, true);
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}

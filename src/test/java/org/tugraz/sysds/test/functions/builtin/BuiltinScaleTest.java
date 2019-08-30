/*
 * Copyright 2019 Graz University of Technology
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

public class BuiltinScaleTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "scale";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinScaleTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-8;
	private final static int rows = 1765;
	private final static int cols = 392;
	private final static double spSparse = 0.7;
	private final static double spDense = 0.1;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}

	@Test
	public void testScaleDenseNegNegCP() {
		runScaleTest(false, false, false, ExecType.CP);
	}
	
	@Test
	public void testScaleDenseNegPosCP() {
		runScaleTest(false, false, true, ExecType.CP);
	}
	
	@Test
	public void testScaleDensePosNegCP() {
		runScaleTest(false, true, false, ExecType.CP);
	}
	
	@Test
	public void testScaleDensePosPosCP() {
		runScaleTest(false, true, true, ExecType.CP);
	}
	
	@Test
	public void testScaleDenseNegNegSP() {
		runScaleTest(false, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testScaleDenseNegPosSP() {
		runScaleTest(false, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testScaleDensePosNegSP() {
		runScaleTest(false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testScaleDensePosPosSP() {
		runScaleTest(false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testScaleSparseNegNegCP() {
		runScaleTest(true, false, false, ExecType.CP);
	}
	
	@Test
	public void testScaleSparseNegPosCP() {
		runScaleTest(true, false, true, ExecType.CP);
	}
	
	@Test
	public void testScaleSparsePosNegCP() {
		runScaleTest(true, true, false, ExecType.CP);
	}
	
	@Test
	public void testScaleSparsePosPosCP() {
		runScaleTest(true, true, true, ExecType.CP);
	}
	
	@Test
	public void testScaleSparseNegNegSP() {
		runScaleTest(true, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testScaleSparseNegPosSP() {
		runScaleTest(true, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testScaleSparsePosNegSP() {
		runScaleTest(true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testScaleSparsePosPosSP() {
		runScaleTest(true, true, true, ExecType.SPARK);
	}
	
	private void runScaleTest(boolean sparse, boolean center, boolean scale, ExecType instType)
	{
		ExecMode platformOld = setExecMode(instType);
		
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", input("A"),
				String.valueOf(center).toUpperCase(), String.valueOf(scale).toUpperCase(), 
				output("B") };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " 
				+ String.valueOf(center).toUpperCase() + " " + String.valueOf(scale).toUpperCase() + 
				" " + expectedDir();
			
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

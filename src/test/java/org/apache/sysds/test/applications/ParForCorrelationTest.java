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

package org.apache.sysds.test.applications;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class ParForCorrelationTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "parfor_corr";
	private final static String TEST_DIR = "applications/parfor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForCorrelationTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 3578;
	private final static int cols1 = 20;      // # of columns in each vector
	
	private final static double minVal=0;    // minimum value in each vector
	private final static double maxVal=1000; // maximum value in each vector

	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Rout" }) );
	}

	@Test
	public void testForCorrleationSerialSerialCP() {
		runParForCorrelationTest(false, PExecMode.LOCAL, PExecMode.LOCAL, ExecType.CP, false, false, false);
	}
	
	@Test
	public void testParForCorrleationLocalLocalCP() {
		runParForCorrelationTest(true, PExecMode.LOCAL, PExecMode.LOCAL, ExecType.CP, false, false, false);
	}
	
	@Test
	public void testParForCorrleationLocalLocalCPWithStats() {
		runParForCorrelationTest(true, PExecMode.LOCAL, PExecMode.LOCAL, ExecType.CP, false, false, true);
	}

	@Test
	public void testParForCorrleationLocalRemoteCP() {
		runParForCorrelationTest(true, PExecMode.LOCAL, PExecMode.REMOTE_SPARK, ExecType.CP, false, false, false);
	}
	
	@Test
	public void testParForCorrleationRemoteLocalCP() {
		runParForCorrelationTest(true, PExecMode.REMOTE_SPARK, PExecMode.LOCAL, ExecType.CP, false, false, false);
	}
	
	@Test
	public void testParForCorrleationRemoteLocalCPWithStats() {
		runParForCorrelationTest(true, PExecMode.REMOTE_SPARK, PExecMode.LOCAL, ExecType.CP, false, false, true);
	}
	

	@Test
	public void testParForCorrleationDefaultCP() 
	{
		runParForCorrelationTest(true, null, null, ExecType.CP, false, false, false);
	}
	
	private void runParForCorrelationTest( boolean parallel, PExecMode outer, PExecMode inner, ExecType instType, boolean profile, boolean debug, boolean statistics )
	{
		//inst exec type, influenced via rows
		ExecMode oldPlatform = rtplatform;
		rtplatform = ExecMode.HYBRID;
		int cols = cols1;
		
		//script
		int scriptNum = -1;
		if( parallel )
		{
			if( inner == PExecMode.REMOTE_SPARK )      scriptNum=2;
			else if( outer == PExecMode.REMOTE_SPARK ) scriptNum=3;
			else if( outer == PExecMode.LOCAL )        scriptNum=1;
			else if( profile )                      scriptNum=5; //optimized with profile
			else if( debug )                        scriptNum=6; //optimized with profile
			else                                    scriptNum=4; //optimized
		}
		else
		{
			scriptNum = 0;
		}
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);
		
		boolean oldStatistics = DMLScript.STATISTICS;
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + scriptNum + ".dml";
		if( statistics ){
			programArgs = new String[]{ "-stats", "-args",
				input("V"), Integer.toString(rows), Integer.toString(cols), output("PearsonR") };
		}
		else {
			programArgs = new String[]{ "-args",
				input("V"), Integer.toString(rows), Integer.toString(cols), output("PearsonR") };
		}
		
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

		long seed = System.nanoTime();
		double[][] V = getRandomMatrix(rows, cols, minVal, maxVal, 1.0, seed);
		writeInputMatrix("V", V, true);

		try {
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1);
			runRScript(true);
		}
		finally {
			DMLScript.STATISTICS = oldStatistics;
			rtplatform = oldPlatform;
		}
		
		//compare matrices
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("PearsonR");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "PearsonR-DML", "PearsonR-R");
	}
}

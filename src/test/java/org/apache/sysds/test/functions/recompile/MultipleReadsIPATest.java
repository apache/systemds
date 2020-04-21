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

package org.apache.sysds.test.functions.recompile;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class MultipleReadsIPATest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "multiple_reads";
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + MultipleReadsIPATest.class.getSimpleName() + "/";
	
	private final static int rows1 = 10;
	private final static int cols1 = 11;
	private final static int rows2 = 20;
	private final static int cols2 = 21;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "X" }) );
	}
	
	
	@Test
	public void testMultipleReadsCPnoIPA() 
	{
		runMultipleReadsTest(ExecType.CP, false);
	}
	
	@Test
	public void testMultipleReadsCPIPA() 
	{
		runMultipleReadsTest(ExecType.CP, true);
	}

	private void runMultipleReadsTest( ExecType et, boolean IPA )
	{	
		ExecMode platformOld = rtplatform;
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",input("X1"),
				Integer.toString(rows1),
				Integer.toString(cols1),
				input("X2"),
				Integer.toString(rows2),
				Integer.toString(cols2),
				output("X") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			rtplatform = ExecMode.HYBRID;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
						
			double[][] X1 = getRandomMatrix(rows1, cols1, -1, 1, 1.0d, 7);
			writeInputMatrix("X1", X1, true);
			double[][] X2 = getRandomMatrix(rows2, cols2, -1, 1, 1.0d, 7);
			writeInputMatrix("X2", X2, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("X");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("X");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
			
		}
	}
	
}
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

package org.apache.sysds.test.functions.rewrite;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class RewriteMatrixMultChainOptTest2 extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteMMChainTest1";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteMatrixMultChainOptTest2.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}));
	}

	@Test
	public void testMMChain1Singlenode() {
		testMMChain(TEST_NAME1, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testMMChain1Hybrid() {
		testMMChain(TEST_NAME1, ExecMode.HYBRID);
	}
	
	@Test
	public void testMMChain1Spark() {
		testMMChain(TEST_NAME1, ExecMode.HYBRID);
	}

	private void testMMChain(String testname, ExecMode et)
	{
		ExecMode etOld = setExecMode(et);
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-args", output("R") };
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(expectedDir());
			
			//execute tests
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, 1e-8, "Stat-DML", "Stat-R");
		}
		finally {
			resetExecMode(etOld);
			Recompiler.reinitRecompiler();
		}
	}
}

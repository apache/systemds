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

package org.apache.sysds.test.functions.data.rand;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class SampleSeedTest extends AutomatedTestBase 
{	
	private final static String TEST_NAME = "SampleSeed";
	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_CLASS_DIR = TEST_DIR + SampleSeedTest.class.getSimpleName() + "/";
	
	private final static int rows = 30;
	private final static int cols = 3;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R" }) );
	}

	@Test
	public void testMatrixVarSeedCP() {
		runSampleSeedTest(TEST_NAME, ExecType.CP);
	}
	
	//FIXME spark different results with different seed
	//(For now we set without replacement, once fix change FALSE to TRUE)
	@Test
	public void testMatrixVarSeedSP() {
		runSampleSeedTest(TEST_NAME, ExecType.SPARK);
	}
	
	private void runSampleSeedTest(String TEST_NAME, ExecType instType) {
		ExecMode platformOld = setExecMode(instType);
		
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", 
				Integer.toString(rows), Integer.toString(cols), output("R") };
			
			//run test
			runTest(true, false, null, -1);
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			Assert.assertEquals(Double.valueOf(55), dmlfile.get(new CellIndex(1,1))); //52 w/ TRUE
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}

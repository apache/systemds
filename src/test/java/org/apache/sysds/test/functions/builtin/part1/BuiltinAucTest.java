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

package org.apache.sysds.test.functions.builtin.part1;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class BuiltinAucTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "auc";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinAucTest.class.getSimpleName() + "/";

	private double eps = 0.01;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	//FIXME missing spark instruction unique
	
	@Test
	public void testPerfectSeparationOrdered() {
		runAucTest(1.0, new double[]{0,0,0,1,1,1},
			new double[]{0.1,0.2,0.3,0.4,0.55,0.56});
	}
	
	@Test
	public void testPerfectSeparationUnordered() {
		runAucTest(1.0, new double[]{0,1,0,1,0,1},
			new double[]{0.1,0.5,0.2,0.55,0.3,0.56});
	}
	
	@Test
	public void testPerfectSeparationUnorderedDups() {
		runAucTest(1.0, new double[]{0,1,0,1,0,1,0,1,0,1,0,1},
			new double[]{0.1,0.5,0.2,0.55,0.3,0.56,0.1,0.5,0.2,0.55,0.3,0.56});
	}

	//selected cases, double checked with R pROC (but not explicitly compared to avoid dependency)
	
	@Test
	public void testMisc1() {
		runAucTest(0.8899, new double[]{0,0,1,0,1,1},
			new double[]{0.1,0.2,0.3,0.4,0.5,0.55});
	}
	
	@Test
	public void testMisc2() {
		runAucTest(0.8899, new double[]{-1,-1,1,-1,1,1},
			new double[]{0.1,0.2,0.3,0.4,0.5,0.55});
	}
	
	@Test
	public void testMisc3() {
		runAucTest(0.75, new double[]{0,0,1,0,1,1,0,1},
			new double[]{0.1,0.2,0.2,0.21,0.7,0.7,0.7,0.7});
	}
	
	@Test
	public void testMisc4() {
		runAucTest(0.6, new double[]{0,0,1,0,1,1,0,1,0},
			new double[]{0.1,0.2,0.2,0.21,0.7,0.7,0.7,0.7,0.9});
	}
	
	@Test
	public void testMisc5() {
		runAucTest(0.6, new double[]{0,0,0,1,0,1,1,0,1},
			new double[]{0.9,0.1,0.2,0.2,0.21,0.7,0.7,0.7,0.7});
	}
	
	@Test
	public void testMisc6() {
		runAucTest(0.5, new double[]{0,0,1,0,1,1,0,1,0,0},
			new double[]{0.1,0.2,0.2,0.21,0.7,0.7,0.7,0.7,0.9,0.9});
	}
	
	@Test
	public void testMisc7() {
		runAucTest(0.4286, new double[]{0,0,1,0,1,1,0,1,0,0,0},
			new double[]{0.1,0.2,0.2,0.21,0.7,0.7,0.7,0.7,0.9,0.9,0.99});
	}
	
	private void runAucTest(double auc, double[] Y, double[] P)
	{
		ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("Yt"), input("Pt"), output("C") };

			//generate actual dataset 
			writeInputMatrixWithMTD("Yt", new double[][]{Y}, false);
			writeInputMatrixWithMTD("Pt", new double[][]{P}, false);

			//execute test
			runTest(true, false, null, -1);

			//compare matrices 
			double val = readDMLMatrixFromOutputDir("C").get(new CellIndex(1,1));
			Assert.assertEquals("Incorrect values: ", auc, val, eps);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}

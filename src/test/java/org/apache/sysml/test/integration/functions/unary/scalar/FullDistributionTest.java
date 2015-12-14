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

package org.apache.sysml.test.integration.functions.unary.scalar;

import java.util.HashMap;
import java.util.Random;

import org.junit.Test;

import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class FullDistributionTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "DFTest";
	
	enum TEST_TYPE { NORMAL, NORMAL_NOPARAMS, NORMAL_MEAN, NORMAL_SD, F, T, CHISQ, EXP, EXP_NOPARAMS };
	
	private final static String TEST_DIR = "functions/unary/scalar/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullDistributionTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "dfout" }));
	}
	
	@Test
	public void testNormal() {
		runDFTest(TEST_TYPE.NORMAL, true, 1.0, 2.0);
	}
	
	@Test
	public void testNormalNoParams() {
		runDFTest(TEST_TYPE.NORMAL_NOPARAMS, true, null, null);
	}
	
	@Test
	public void testNormalMean() {
		runDFTest(TEST_TYPE.NORMAL_MEAN, true, 1.0, null);
	}
	
	@Test
	public void testNormalSd() {
		runDFTest(TEST_TYPE.NORMAL_SD, true, 2.0, null);
	}
	
	@Test
	public void testT() {
		runDFTest(TEST_TYPE.T, true, 10.0, null);
	}
	
	@Test
	public void testF() {
		runDFTest(TEST_TYPE.T, true, 10.0, 20.0);
	}
	
	@Test
	public void testChisq() {
		runDFTest(TEST_TYPE.CHISQ, true, 10.0, null);
	}
	
	@Test
	public void testExp() {
		runDFTest(TEST_TYPE.EXP, true, 5.0, null);
	}
	
	private void runDFTest(TEST_TYPE type, boolean inverse, Double param1, Double param2) {
		getAndLoadTestConfiguration(TEST_NAME);

		double in = (new Random(System.nanoTime())).nextDouble();
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + "_" + type.toString() + ".dml";
		fullRScriptName = HOME + TEST_NAME + "_" + type.toString() + ".R";
		
		String DMLout = output("dfout");
		String Rout = expected("dfout");
		
		switch(type) {
		case NORMAL_NOPARAMS:
			programArgs = new String[]{"-args", Double.toString(in), DMLout };
			rCmd = "Rscript" + " " + fullRScriptName + " " + Double.toString(in) + " " + Rout;
			break;
			
		case NORMAL_MEAN:
		case NORMAL_SD:
		case T:
		case CHISQ:
		case EXP:
			programArgs = new String[]{"-args", Double.toString(in), Double.toString(param1), DMLout };
			rCmd = "Rscript" + " " + fullRScriptName + " " + Double.toString(in) + " " + Double.toString(param1) + " " + Rout;
			break;
			
		case NORMAL:
		case F:
			programArgs = new String[]{"-args", Double.toString(in), Double.toString(param1), Double.toString(param2), DMLout };
			rCmd = "Rscript" + " " + fullRScriptName + " " + Double.toString(in) + " " + Double.toString(param1) + " " + Double.toString(param2) + " " + Rout;
			break;
		
			default: 
				throw new RuntimeException("Invalid distribution function: " + type);
		}
		
		runTest(true, false, null, -1); 
		runRScript(true); 
		
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("dfout");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("dfout");
		TestUtils.compareMatrices(dmlfile, rfile, 1e-8, "DMLout", "Rout");

	}
	
}

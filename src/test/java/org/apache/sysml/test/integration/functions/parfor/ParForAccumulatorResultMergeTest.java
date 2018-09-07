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

package org.apache.sysml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class ParForAccumulatorResultMergeTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_NAME1 = "parfor_accumulator1"; //local merge
	private final static String TEST_NAME2 = "parfor_accumulator2"; //remote MR merge
	private final static String TEST_NAME3 = "parfor_accumulator3"; //remote SPARK merge
	
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForAccumulatorResultMergeTest.class.getSimpleName() + "/";
	
	private final static double eps = 0;
	private final static int rows = 1210;
	private final static int cols = 345;
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }) );
	}

	@Test
	public void testParForAccumulatorLocalEmptyDense() {
		runParForAccumulatorResultMergeTest(TEST_NAME1, false, false, ExecType.CP);
	}
	
	@Test
	public void testParForAccumulatorLocalEmptySparse() {
		runParForAccumulatorResultMergeTest(TEST_NAME1, false, true, ExecType.CP);
	}
	
	@Test
	public void testParForAccumulatorLocalInitDense() {
		runParForAccumulatorResultMergeTest(TEST_NAME1, true, false, ExecType.CP);
	}
	
	@Test
	public void testParForAccumulatorLocalInitSparse() {
		runParForAccumulatorResultMergeTest(TEST_NAME1, true, true, ExecType.CP);
	}
	
	@Test
	public void testParForAccumulatorRemoteEmptyDenseMR() {
		runParForAccumulatorResultMergeTest(TEST_NAME2, false, false, ExecType.MR);
	}
	
	@Test
	public void testParForAccumulatorRemoteEmptySparseMR() {
		runParForAccumulatorResultMergeTest(TEST_NAME2, false, true, ExecType.MR);
	}
	
	@Test
	public void testParForAccumulatorRemoteInitDenseMR() {
		runParForAccumulatorResultMergeTest(TEST_NAME2, true, false, ExecType.MR);
	}
	
	@Test
	public void testParForAccumulatorRemoteInitSparseMR() {
		runParForAccumulatorResultMergeTest(TEST_NAME2, true, true, ExecType.MR);
	}
	
	@Test
	public void testParForAccumulatorRemoteEmptyDenseSP() {
		runParForAccumulatorResultMergeTest(TEST_NAME3, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testParForAccumulatorRemoteEmptySparseSP() {
		runParForAccumulatorResultMergeTest(TEST_NAME3, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testParForAccumulatorRemoteInitDenseSP() {
		runParForAccumulatorResultMergeTest(TEST_NAME3, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testParForAccumulatorRemoteInitSparseSP() {
		runParForAccumulatorResultMergeTest(TEST_NAME3, true, true, ExecType.SPARK);
	}

	private void runParForAccumulatorResultMergeTest( String test, boolean init, boolean sparse, ExecType et )
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ) {
			case CP: rtplatform = RUNTIME_PLATFORM.SINGLE_NODE; break;
			case MR: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK; break;
			default: throw new RuntimeException("Unsupported exec type: "+et.name());
		}
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( et == ExecType.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try {
			String TEST_NAME = test;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", 
				String.valueOf(rows), String.valueOf(cols), String.valueOf(init).toUpperCase(),
				String.valueOf(sparse).toUpperCase(), output("R") };
			fullRScriptName = HOME + TEST_NAME.substring(0, TEST_NAME.length()-1) + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				String.valueOf(rows) + " " + String.valueOf(cols) + " " + String.valueOf(init).toUpperCase() 
					+ " " + String.valueOf(sparse).toUpperCase() + " " + expectedDir();
			
			//run tests
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}

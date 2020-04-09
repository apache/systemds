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

package org.apache.sysds.test.functions.misc;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class ZeroRowsColsMatrixTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "ZeroMatrix_RemoveEmpty";
	private final static String TEST_NAME2 = "ZeroMatrix_Cbind";
	private final static String TEST_NAME3 = "ZeroMatrix_Rbind";
	private final static String TEST_NAME4 = "ZeroMatrix_Aggregates";
	
	private final static String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ZeroRowsColsMatrixTest.class.getSimpleName() + "/";
	private final static int dim = 1372;
	private final static double eps = 1e-8;
	
	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" })); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" })); 
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" })); 
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "R" })); 
	}
	
	@Test
	public void testEmptyMatrixRemoveEmptyNoRewritesEmptyRetCP() {
		runEmptyMatrixTest(TEST_NAME1, false, true, ExecType.CP);
	}
	
	@Test
	public void testEmptyMatrixRemoveEmptyRewritesEmptyRetCP() {
		runEmptyMatrixTest(TEST_NAME1, true, true, ExecType.CP);
	}
	
	@Test
	public void testEmptyMatrixRemoveEmptyNoRewritesEmptyRetSP() {
		runEmptyMatrixTest(TEST_NAME1, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testEmptyMatrixRemoveEmptyRewritesEmptyRetSP() {
		runEmptyMatrixTest(TEST_NAME1, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testEmptyMatrixRemoveEmptyNoRewritesCP() {
		runEmptyMatrixTest(TEST_NAME1, false, false, ExecType.CP);
	}
	
	@Test
	public void testEmptyMatrixRemoveEmptyRewritesCP() {
		runEmptyMatrixTest(TEST_NAME1, true, false, ExecType.CP);
	}
	
	
	@Test
	public void testEmptyMatrixRemoveEmptyNoRewritesSP() {
		runEmptyMatrixTest(TEST_NAME1, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testEmptyMatrixRemoveEmptyRewritesSP() {
		runEmptyMatrixTest(TEST_NAME1, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testEmptyMatrixCbindNoRewritesCP() {
		runEmptyMatrixTest(TEST_NAME2, false, ExecType.CP);
	}
	
	@Test
	public void testEmptyMatrixCbindRewritesCP() {
		runEmptyMatrixTest(TEST_NAME2, true, ExecType.CP);
	}
	
	@Test
	public void testEmptyMatrixCbindNoRewritesSP() {
		runEmptyMatrixTest(TEST_NAME2, false, ExecType.SPARK);
	}
	
	@Test
	public void testEmptyMatrixCbindRewritesSP() {
		runEmptyMatrixTest(TEST_NAME2, true, ExecType.SPARK);
	}
	
	@Test
	public void testEmptyMatrixRbindNoRewritesCP() {
		runEmptyMatrixTest(TEST_NAME3, false, ExecType.CP);
	}
	
	@Test
	public void testEmptyMatrixRbindRewritesCP() {
		runEmptyMatrixTest(TEST_NAME3, true, ExecType.CP);
	}
	
	@Test
	public void testEmptyMatrixRbindNoRewritesSP() {
		runEmptyMatrixTest(TEST_NAME3, false, ExecType.SPARK);
	}
	
	@Test
	public void testEmptyMatrixRbindRewritesSP() {
		runEmptyMatrixTest(TEST_NAME3, true, ExecType.SPARK);
	}
	
	@Test
	public void testEmptyMatrixAggregatesNoRewritesCP() {
		runEmptyMatrixTest(TEST_NAME4, false, ExecType.CP);
	}
	
	@Test
	public void testEmptyMatrixAggregatesRewritesCP() {
		runEmptyMatrixTest(TEST_NAME4, true, ExecType.CP);
	}
	
	@Test
	public void testEmptyMatrixAggregatesNoRewritesSP() {
		runEmptyMatrixTest(TEST_NAME4, false, ExecType.SPARK);
	}
	
	@Test
	public void testEmptyMatrixAggregatesRewritesSP() {
		runEmptyMatrixTest(TEST_NAME4, true, ExecType.SPARK);
	}
	
	private void runEmptyMatrixTest( String testname, boolean rewrites, ExecType et ) {
		runEmptyMatrixTest(testname, rewrites, false, et);
	}
	
	private void runEmptyMatrixTest( String testname, boolean rewrites, boolean emptyRet, ExecType et )
	{
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		
		try {
			String TEST_NAME = testname;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", String.valueOf(dim),
				String.valueOf(emptyRet).toUpperCase(), output("R")};
			
			fullRScriptName = HOME + TEST_NAME +".R";
			rCmd = getRCmd(String.valueOf(dim),
				String.valueOf(emptyRet).toUpperCase(), expectedDir());
	
			//run Tests
			runTest(true, false, null, -1); 
			runRScript(true);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check meta data
			if( !testname.equals(TEST_NAME4) )
				checkDMLMetaDataFile("R", new MatrixCharacteristics(dim, 3, 1000, 1000));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}

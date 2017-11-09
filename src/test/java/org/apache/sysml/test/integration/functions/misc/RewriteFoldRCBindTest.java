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

package org.apache.sysml.test.integration.functions.misc;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

public class RewriteFoldRCBindTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteFoldCBind";
	private static final String TEST_NAME2 = "RewriteFoldRBind";
	
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteFoldRCBindTest.class.getSimpleName() + "/";
	
	private static final int rows = 1932;
	private static final int cols = 14;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
	}

	@Test
	public void testRewriteFoldCBindNoRewrite() {
		testRewriteFoldRCBind( TEST_NAME1, false, ExecType.CP );
	}
	
	@Test
	public void testRewriteFoldCBindRewrite() {
		testRewriteFoldRCBind( TEST_NAME1, true, ExecType.CP );
	}
	
	@Test
	public void testRewriteFoldRBindNoRewrite() {
		testRewriteFoldRCBind( TEST_NAME2, false, ExecType.CP );
	}
	
	@Test
	public void testRewriteFoldRBindRewrite() {
		testRewriteFoldRCBind( TEST_NAME2, true, ExecType.CP );
	}

	private void testRewriteFoldRCBind( String testname, boolean rewrites, ExecType et )
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats", "-args", String.valueOf(rows), 
					String.valueOf(cols), output("R") };
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			//run performance tests
			runTest(true, false, null, -1); 
			
			//compare matrices 
			Double ret = readDMLMatrixFromHDFS("R").get(new CellIndex(1,1));
			Assert.assertEquals("Wrong result", new Double(5*rows*cols), ret);
			
			//check for applied rewrites
			if( rewrites ) {
				Assert.assertTrue(!heavyHittersContainsString("append")
					&& Statistics.getCPHeavyHitterCount("cbind") <= 1
					&& Statistics.getCPHeavyHitterCount("rbind") <= 1);
			}
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			rtplatform = platformOld;
		}
	}
}

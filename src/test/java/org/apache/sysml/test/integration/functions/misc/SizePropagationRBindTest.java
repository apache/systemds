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

import org.junit.Test;

import org.junit.Assert;

import java.util.HashMap;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class SizePropagationRBindTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "SizePropagationRBind";
	
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + SizePropagationRBindTest.class.getSimpleName() + "/";
	
	private static final int N = 100;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
	}

	@Test
	public void testSizePropagationRBindNoRewrites() {
		testSizePropagationRBind( TEST_NAME1, false );
	}
	
	@Test
	public void testSizePropagationRBindRewrites() {
		testSizePropagationRBind( TEST_NAME1, true );
	}
	
	private void testSizePropagationRBind( String testname, boolean rewrites ) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		RUNTIME_PLATFORM oldPlatform = rtplatform;
		
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats","-args", String.valueOf(N), output("R") };
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
			
			runTest(true, false, null, -1); 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			Assert.assertTrue( dmlfile.get(new CellIndex(1,1))==2*(N+2) );
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			rtplatform = oldPlatform;
		}
	}
}
